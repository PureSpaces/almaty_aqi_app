import json
import math
import os
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template_string
import folium
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)

# Configuration & Secrets
API_KEY = os.environ.get("AAI_API_KEY")
CSV_FILE = "almaty_pm25_data.csv"
FACTORY_CSV_FILE = "factory_pm25_predictions.csv"
API_URL = "https://backend.air.org.kz/v1/measurements/map"

ALMATY_TZ = timezone(timedelta(hours=5))

# Almaty Spatial Bounds
ALMATY_SW = [43.10, 76.70]
ALMATY_NE = [43.40, 77.15]

# Factory & Industrial Zones
INDUSTRIAL_ZONES = [
    {
        "name": "CHP-2 Power Plant Zone (TETS-2)",
        "lat": 43.2625,
        "lon": 76.7950,
        "radius_m": 8000,
        "description": "Major coal-fired power station & western industrial corridor",
    },
    {
        "name": "CHP-1 Power Plant Zone (TETS-1)",
        "lat": 43.2780,
        "lon": 76.9280,
        "radius_m": 5000,
        "description": "Central heat & power station and rail industrial belt",
    },
    {
        "name": "Alatau Industrial Zone",
        "lat": 43.3200,
        "lon": 76.8300,
        "radius_m": 7000,
        "description": "Northern manufacturing, logistics, and heavy industrial park",
    },
]


def get_almaty_now():
    return datetime.now(timezone.utc).astimezone(ALMATY_TZ)


def parse_timestamp(ts_str):
    if not ts_str:
        return None, None
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        dt_almaty = dt.astimezone(ALMATY_TZ)
        return dt_almaty.strftime("%Y-%m-%d"), dt_almaty.strftime("%H:00")
    except (ValueError, TypeError):
        return None, None


def get_aqi_info(pm25):
    if pm25 is None or pd.isna(pm25):
        return "#808080", "#FFFFFF", "No Data", "128,128,128"
    elif pm25 <= 12.0:
        return "#00E400", "#000000", "Good", "0,228,0"
    elif pm25 <= 35.4:
        return "#FFFF00", "#000000", "Moderate", "255,255,0"
    elif pm25 <= 55.4:
        return "#FF7E00", "#FFFFFF", "Sensitive Groups", "255,126,0"
    elif pm25 <= 150.4:
        return "#FF0000", "#FFFFFF", "Unhealthy", "255,0,0"
    elif pm25 <= 250.4:
        return "#8F3F97", "#FFFFFF", "Very Unhealthy", "143,63,151"
    else:
        return "#7E0023", "#FFFFFF", "Hazardous", "126,0,35"


def extract_pm25_data(data):
    records = []
    payload = data.get("data", []) if isinstance(data, dict) else data
    items = payload if isinstance(payload, list) else [payload]

    for item in items:
        if not isinstance(item, dict):
            continue

        station_name = (
            item.get("location_name")
            or item.get("name")
            or item.get("title")
            or "Unknown Station"
        )
        pm25 = (
            item.get("pm25_avg")
            or item.get("pm25")
            or item.get("pm2_5")
            or item.get("value")
        )
        try:
            pm25 = float(pm25) if pm25 is not None else None
        except (ValueError, TypeError):
            pm25 = None

        raw_time = (
            item.get("hour")
            or item.get("last_measurement")
            or item.get("timestamp")
            or item.get("updated_at")
        )
        date_str, hour_str = parse_timestamp(raw_time)

        lat = item.get("lat") or item.get("latitude")
        lon = item.get("lon") or item.get("lng") or item.get("longitude")

        if (lat is None or lon is None) and isinstance(item.get("location"), dict):
            loc = item["location"]
            lat = lat or loc.get("lat") or loc.get("latitude")
            lon = lon or loc.get("lon") or loc.get("lng") or loc.get("longitude")
        elif (lat is None or lon is None) and isinstance(item.get("coordinates"), dict):
            coords = item["coordinates"]
            lat = lat or coords.get("lat") or coords.get("latitude")
            lon = lon or coords.get("lon") or coords.get("lng") or coords.get("longitude")
        elif (lat is None or lon is None) and isinstance(item.get("coordinates"), list) and len(item["coordinates"]) >= 2:
            lat = item["coordinates"][1]
            lon = item["coordinates"][0]

        try:
            lat = float(lat) if lat is not None else None
            lon = float(lon) if lon is not None else None
        except (ValueError, TypeError):
            lat, lon = None, None

        records.append({
            "date": date_str,
            "hour": hour_str,
            "station_name": station_name,
            "pm2.5": pm25,
            "latitude": lat,
            "longitude": lon,
        })
    return records


def fetch_and_update_data():
    """Fetches live API data and appends to local historical CSV."""
    if not API_KEY:
        print("⚠ Warning: AAI_API_KEY environment variable is missing. Skipping API fetch.")
        return

    headers = {"x-api-key": API_KEY, "Accept": "application/json"}
    try:
        response = requests.get(API_URL, headers=headers, timeout=15)
        response.raise_for_status()

        records = extract_pm25_data(response.json())
        new_df = pd.DataFrame(records)

        if new_df.empty:
            print("⚠ API returned empty dataset.")
            return

        if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
            existing_df = pd.read_csv(CSV_FILE)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(
                subset=["date", "hour", "station_name"],
                keep="last",
                inplace=True,
            )
        else:
            combined_df = new_df

        combined_df.to_csv(CSV_FILE, index=False)
        print(f"✅ Historical dataset updated: {len(combined_df)} total records stored.")

        # Log spatial industrial predictions for new hourly slice
        log_factory_predictions(combined_df)

    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Failed: {e}")


def log_factory_predictions(df):
    """Generates and appends radial industrial zone predictions per hour."""
    geo_df = df.dropna(subset=["latitude", "longitude", "pm2.5"]).copy()
    if geo_df.empty:
        return

    latest_time = geo_df.sort_values(by=["date", "hour"]).iloc[-1]
    date_str, hour_str = latest_time["date"], latest_time["hour"]

    hour_df = geo_df[(geo_df["date"] == date_str) & (geo_df["hour"] == hour_str)]
    known_stations = [
        {"lat": float(r["latitude"]), "lon": float(r["longitude"]), "pm25": float(r["pm2.5"])}
        for _, r in hour_df.iterrows()
    ]

    records = []
    for factory in INDUSTRIAL_ZONES:
        max_r_km = factory["radius_m"] / 1000.0
        for r_km in np.arange(0, max_r_km + 0.1, 1.0):
            num_angles = 1 if r_km == 0 else max(8, int(math.pi * 2 * r_km))
            for angle in np.linspace(0, 360, num_angles, endpoint=False):
                rad = math.radians(angle)
                s_lat = factory["lat"] + (r_km * math.cos(rad)) / 111.0
                s_lon = factory["lon"] + (r_km * math.sin(rad)) / 81.0

                # IDW Prediction
                num, den = 0.0, 0.0
                for st in known_stations:
                    d = math.sqrt((s_lat - st["lat"]) ** 2 + (s_lon - st["lon"]) ** 2)
                    if d < 0.0001:
                        pred_pm25 = st["pm25"]
                        break
                    w = 1.0 / (d ** 2.0)
                    num += w * st["pm25"]
                    den += w
                else:
                    pred_pm25 = num / den if den != 0 else 0.0

                _, _, status, _ = get_aqi_info(pred_pm25)
                records.append({
                    "date": date_str,
                    "hour": hour_str,
                    "factory_name": factory["name"],
                    "sample_lat": round(s_lat, 5),
                    "sample_lon": round(s_lon, 5),
                    "distance_km": round(r_km, 2),
                    "predicted_pm25": round(pred_pm25, 2),
                    "aqi_status": status,
                })

    pred_df = pd.DataFrame(records)
    if os.path.exists(FACTORY_CSV_FILE) and os.path.getsize(FACTORY_CSV_FILE) > 0:
        existing_pred = pd.read_csv(FACTORY_CSV_FILE)
        combined_pred = pd.concat([existing_pred, pred_df], ignore_index=True)
        combined_pred.drop_duplicates(
            subset=["date", "hour", "factory_name", "sample_lat", "sample_lon"],
            keep="last",
            inplace=True,
        )
    else:
        combined_pred = pred_df

    combined_pred.to_csv(FACTORY_CSV_FILE, index=False)


def generate_map_html():
    """Builds the interactive map HTML with full historical timeline scrubbing capabilities."""
    if not os.path.exists(CSV_FILE):
        return "<h3>No historical data found. Please run the data fetcher.</h3>"

    df = pd.read_csv(CSV_FILE).dropna(subset=["latitude", "longitude", "pm2.5"])
    df["timestamp_key"] = df["date"] + " " + df["hour"]

    # Group records by hourly timestamps
    timestamps = sorted(df["timestamp_key"].unique())
    timeline_data = {}

    for ts in timestamps:
        slice_df = df[df["timestamp_key"] == ts]
        timeline_data[ts] = [
            {
                "name": str(r["station_name"]),
                "lat": float(r["latitude"]),
                "lon": float(r["longitude"]),
                "pm25": float(r["pm2.5"]),
            }
            for _, r in slice_df.iterrows()
        ]

    m = folium.Map(
        location=[43.2389, 76.8897],
        zoom_start=12,
        tiles=None,
        zoom_control=True,
    )

    # Industrial Zone Overlays
    factory_layer = folium.FeatureGroup(name="Industrial Zones").add_to(m)
    for zone in INDUSTRIAL_ZONES:
        folium.Circle(
            location=[zone["lat"], zone["lon"]],
            radius=zone["radius_m"],
            color="#00E5FF",
            weight=2.5,
            dash_array="6, 6",
            fill=True,
            fill_color="#00E5FF",
            fill_opacity=0.12,
            tooltip=f"🏭 Industrial Zone: {zone['name']}",
        ).add_to(factory_layer)

    js_timeline_json = json.dumps(timeline_data)
    js_timestamps_json = json.dumps(timestamps)

    custom_html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
        .leaflet-container {{ font-family: 'Plus Jakarta Sans', sans-serif !important; background-color: #0f172a; }}
        .theme-dark .leaflet-tile-pane {{ filter: brightness(0.75) contrast(1.2) saturate(0.8); }}

        .custom-pm25-badge {{
            background: transparent !important;
            border: none !important;
        }}

        .badge-core {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: -0.2px;
            border: 2.5px solid #ffffff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}

        #ui-control-card {{
            position: fixed; top: 16px; right: 16px; width: 300px; border-radius: 14px; padding: 16px;
            z-index: 9999; backdrop-filter: blur(12px); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        .theme-light #ui-control-card {{ background: rgba(255, 255, 255, 0.92); color: #1e293b; }}
        .theme-dark #ui-control-card {{ background: rgba(15, 23, 42, 0.92); color: #f8fafc; border-color: rgba(255, 255, 255, 0.12); }}
        .time-slider {{ width: 100%; accent-color: #3b82f6; cursor: pointer; }}
    </style>

    <div id="ui-control-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <b style="font-size:15px;">Almaty AQI Dashboard</b>
            <span id="theme-icon">☀️</span>
        </div>
        <div style="font-size:11px; margin-bottom:10px; opacity:0.8;">Live & Historical ML Timeline</div>

        <div style="padding:8px; border-radius:8px; background:rgba(59,130,246,0.1); margin-bottom:10px; font-size:12px; font-weight:700;">
            📅 Selected Hour: <span id="active-time-label" style="color:#3b82f6;">Loading...</span>
        </div>

        <div>
            <label style="font-size:11px; font-weight:700;">Historical Timeline Scrubbing</label>
            <input type="range" min="0" max="{max(0, len(timestamps) - 1)}" value="{max(0, len(timestamps) - 1)}" id="timeline-slider" class="time-slider" oninput="onTimelineChange(this.value)">
        </div>
    </div>

    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var map = null;
        for (var key in window) {{
            if (key.startsWith("map_") && window[key] instanceof L.Map) {{
                map = window[key]; break;
            }}
        }}
        if (!map) return;

        var esriStreet = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}");
        var esriDark = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}");
        esriStreet.addTo(map);

        var timelineData = {js_timeline_json};
        var timestamps = {js_timestamps_json};
        var activeMarkers = [];
        var currentStations = [];

        function renderHour(index) {{
            var tsKey = timestamps[index];
            document.getElementById("active-time-label").innerText = tsKey;
            currentStations = timelineData[tsKey] || [];

            // Clear previous station markers
            activeMarkers.forEach(m => map.removeLayer(m));
            activeMarkers = [];

            currentStations.forEach(st => {{
                var catColor = getAQIColor(st.pm25);
                var textColor = getTextColor(st.pm25);
                var pmValStr = st.pm25.toFixed(1);

                var badgeIcon = L.divIcon({{
                    className: 'custom-pm25-badge',
                    html: '<div class="badge-core" style="background-color:' + catColor + '; color:' + textColor + ';">' + pmValStr + '</div>',
                    iconSize: [36, 36],
                    iconAnchor: [18, 18]
                }});

                var marker = L.marker([st.lat, st.lon], {{ icon: badgeIcon }})
                    .bindTooltip(st.name + ": " + pmValStr + " µg/m³");
                
                marker.addTo(map);
                activeMarkers.push(marker);
            }});
        }}

        function getAQIColor(val) {{
            if (val <= 12.0) return "#00E400";
            if (val <= 35.4) return "#FFFF00";
            if (val <= 55.4) return "#FF7E00";
            if (val <= 150.4) return "#FF0000";
            if (val <= 250.4) return "#8F3F97";
            return "#7E0023";
        }}

        function getTextColor(val) {{
            if (val <= 35.4) return "#000000";
            return "#FFFFFF";
        }}

        window.onTimelineChange = function(idx) {{
            renderHour(parseInt(idx, 10));
        }};

        // Hover AI Prediction Engine
        var tooltipDiv = document.createElement("div");
        tooltipDiv.style.position = "absolute"; tooltipDiv.style.display = "none";
        tooltipDiv.style.zIndex = "10000"; tooltipDiv.style.background = "rgba(15,23,42,0.94)";
        tooltipDiv.style.color = "#fff"; tooltipDiv.style.padding = "8px 12px";
        tooltipDiv.style.borderRadius = "8px"; tooltipDiv.style.pointerEvents = "none";
        document.body.appendChild(tooltipDiv);

        map.on("mousemove", function(e) {{
            var lat = e.latlng.lat, lng = e.latlng.lng;
            if (lat < 43.10 || lat > 43.40 || lng < 76.70 || lng > 77.15) {{
                tooltipDiv.style.display = "none"; return;
            }}
            var predVal = predictIDW(lat, lng, currentStations);
            tooltipDiv.innerHTML = "<b>🤖 AI Spatial Prediction</b><br>" + predVal.toFixed(1) + " µg/m³";
            tooltipDiv.style.left = (e.originalEvent.pageX + 15) + "px";
            tooltipDiv.style.top = (e.originalEvent.pageY + 15) + "px";
            tooltipDiv.style.display = "block";
        }});

        function predictIDW(lat, lng, stations) {{
            if (!stations || stations.length === 0) return 0.0;
            var num = 0.0, den = 0.0;
            for (var i = 0; i < stations.length; i++) {{
                var d = Math.sqrt(Math.pow(lat - stations[i].lat, 2) + Math.pow(lng - stations[i].lon, 2));
                if (d < 0.0001) return stations[i].pm25;
                var w = 1.0 / Math.pow(d, 2.0);
                num += w * stations[i].pm25; den += w;
            }}
            return den === 0 ? 0.0 : num / den;
        }}

        if (timestamps.length > 0) renderHour(timestamps.length - 1);
    }});
    </script>
    """

    m.get_root().html.add_child(folium.Element(custom_html))
    return m._repr_html_()


@app.route("/")
def index():
    return render_template_string(generate_map_html())


@app.route("/cron/fetch", methods=["GET", "POST"])
def cron_trigger():
    fetch_and_update_data()
    return {"status": "success", "message": "Hourly data updated successfully."}


if __name__ == "__main__":
    # 1. Fetch latest data and update CSVs
    fetch_and_update_data()
    
    # 2. Build and save index.html
    map_html = generate_map_html()
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(map_html)
    print("✅ index.html generated successfully!")

    # 3. ONLY run Flask locally (GitHub Actions skips this block)
    if not os.environ.get("GITHUB_ACTIONS"):
        app.run(host="127.0.0.1", port=5000, debug=True)
