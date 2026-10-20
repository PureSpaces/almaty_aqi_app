import json
import os
from datetime import datetime, timedelta, timezone
import folium
import pandas as pd
import requests

# Config & File Paths
API_KEY = ''
CSV_FILE = "almaty_pm25_data.csv"
MAP_FILE = "almaty_pm25_map.html"
API_URL = "https://backend.air.org.kz/v1/measurements/map"

# Almaty Timezone Constant (UTC+5)
ALMATY_TZ = timezone(timedelta(hours=5))


def get_almaty_now():
    """Returns current datetime in Almaty timezone (UTC+5)."""
    return datetime.now(timezone.utc).astimezone(ALMATY_TZ)


def parse_timestamp(ts_str):
    """Splits ISO timestamp string into separate date (YYYY-MM-DD) and hour (HH:00) strings."""
    if not ts_str:
        return None, None
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        dt_almaty = dt.astimezone(ALMATY_TZ)
        return dt_almaty.strftime("%Y-%m-%d"), dt_almaty.strftime("%H:00")
    except (ValueError, TypeError):
        return None, None


def get_aqi_info(pm25):
    """
    Returns (color_hex, text_color_hex, status_label, rgb_tuple_str)
    based on US EPA PM2.5 standards.
    """
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
    """
    Parses response payload from backend.air.org.kz/v1/measurements/map.
    Extracts station metadata, coordinates, timestamp, and PM2.5 concentration.
    """
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

        records.append(
            {
                "date": date_str,
                "hour": hour_str,
                "station_name": station_name,
                "pm2.5": pm25,
                "latitude": lat,
                "longitude": lon,
            }
        )

    return records


def render_folium_map(df, output_html=MAP_FILE):
    """
    Renders an interactive Folium map using API-key-free Esri & OpenStreetMap tile servers,
    complete with real-time Almaty clock, day/night theme control, PM2.5 markers, and spatial ML interpolation.
    """
    geo_df = df.dropna(subset=["latitude", "longitude"]).copy()

    if geo_df.empty:
        print("⚠️ No valid station coordinates found to draw on the map.")
        return

    center_lat = geo_df["latitude"].mean() if not geo_df.empty else 43.2389
    center_lon = geo_df["longitude"].mean() if not geo_df.empty else 76.8897

    # Initialize map without built-in tiles (JS manager initializes reliable free tile layers)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=None,
        zoom_control=True,
    )

    latest_stations = (
        geo_df.sort_values(by=["date", "hour"], ascending=True)
        .groupby("station_name")
        .last()
        .reset_index()
    )

    stations_list = []

    for idx, row in latest_stations.iterrows():
        pm25 = row["pm2.5"]
        color, text_color, status, rgb_str = get_aqi_info(pm25)
        pm_val_str = f"{pm25:.1f}" if pm25 is not None else "N/A"

        if pm25 is not None and not pd.isna(pm25):
            stations_list.append(
                {
                    "name": str(row["station_name"]),
                    "lat": float(row["latitude"]),
                    "lon": float(row["longitude"]),
                    "pm25": float(pm25),
                }
            )

        popup_content = f"""
        <div style="font-family: system-ui, -apple-system, sans-serif; font-size: 13px; line-height: 1.5; min-width: 200px;">
            <h4 style="margin: 0 0 6px 0; color: #111827; border-bottom: 2px solid {color}; padding-bottom: 4px;">{row['station_name']}</h4>
            <b>PM2.5:</b> <span style="font-size:14px; font-weight:bold;">{pm_val_str} µg/m³</span><br>
            <b>Status:</b> <span style="color:{color}; font-weight:bold; background:rgba({rgb_str},0.15); padding:2px 6px; border-radius:4px;">{status}</span><br>
            <b>Last Updated:</b> {row['date']} {row['hour']}<br>
            <b>Coords:</b> {row['latitude']:.4f}, {row['longitude']:.4f}
        </div>
        """

        flashy_icon_html = f"""
        <div class="flashy-station-marker" style="--aqi-color: {color}; --aqi-rgb: {rgb_str};">
            <div class="pulse-halo"></div>
            <div class="badge-core" style="background-color: {color}; color: {text_color};">
                {pm_val_str}
            </div>
        </div>
        """

        folium.map.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.DivIcon(
                html=flashy_icon_html,
                icon_size=(48, 48),
                icon_anchor=(24, 24),
            ),
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"{row['station_name']}: {pm_val_str} µg/m³ ({status})",
        ).add_to(m)

    js_known_data = json.dumps(stations_list)

    custom_assets_html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

        .leaflet-container {{
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
            background-color: #0f172a;
        }}

        /* Dark Mode Tile Inversion Fallback */
        .theme-dark .leaflet-tile-pane {{
            filter: brightness(0.75) contrast(1.2) saturate(0.8);
        }}

        .flashy-station-marker {{
            position: relative;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }}

        .pulse-halo {{
            position: absolute;
            top: 50%;
            left: 50%;
            width: 42px;
            height: 42px;
            margin-top: -21px;
            margin-left: -21px;
            border-radius: 50%;
            background: rgba(var(--aqi-rgb), 0.4);
            animation: haloPulse 2.2s infinite ease-in-out;
            pointer-events: none;
        }}

        @keyframes haloPulse {{
            0% {{
                transform: scale(0.8);
                opacity: 0.9;
                box-shadow: 0 0 0 0 rgba(var(--aqi-rgb), 0.7);
            }}
            70% {{
                transform: scale(1.5);
                opacity: 0;
                box-shadow: 0 0 18px 12px rgba(var(--aqi-rgb), 0);
            }}
            100% {{
                transform: scale(0.8);
                opacity: 0;
            }}
        }}

        .badge-core {{
            position: relative;
            z-index: 2;
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
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35), 0 0 12px rgba(var(--aqi-rgb), 0.6);
            transition: transform 0.2s ease;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }}

        .flashy-station-marker:hover .badge-core {{
            transform: scale(1.2);
        }}

        #ui-control-card {{
            position: fixed;
            top: 16px;
            right: 16px;
            width: 280px;
            border-radius: 14px;
            padding: 16px;
            z-index: 9999;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .theme-light #ui-control-card {{
            background: rgba(255, 255, 255, 0.92);
            color: #1e293b;
        }}
        .theme-light .card-subtext {{ color: #64748b; }}
        .theme-light .pill-bg {{ background: #f1f5f9; border: 1px solid #e2e8f0; }}

        .theme-dark #ui-control-card {{
            background: rgba(15, 23, 42, 0.92);
            color: #f8fafc;
            border-color: rgba(255, 255, 255, 0.12);
        }}
        .theme-dark .card-subtext {{ color: #94a3b8; }}
        .theme-dark .pill-bg {{ background: #1e293b; border: 1px solid #334155; }}

        .clock-badge {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 12px;
        }}

        .mode-btn-group {{
            display: flex;
            gap: 4px;
            margin-bottom: 12px;
            background: rgba(0, 0, 0, 0.08);
            padding: 3px;
            border-radius: 8px;
        }}
        .mode-btn {{
            flex: 1;
            border: none;
            padding: 6px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            background: transparent;
            color: inherit;
            opacity: 0.7;
        }}
        .mode-btn.active {{
            background: #3b82f6;
            color: #ffffff;
            opacity: 1;
            box-shadow: 0 2px 6px rgba(59, 130, 246, 0.4);
        }}

        .time-slider {{
            width: 100%;
            accent-color: #3b82f6;
            cursor: pointer;
        }}

        .legend-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11px;
            margin-bottom: 4px;
            font-weight: 600;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }}
    </style>

    <div id="ui-control-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <b style="font-size: 15px;">Almaty AQI Dashboard</b>
            <span id="day-night-icon" style="font-size: 18px;">☀️</span>
        </div>
        <div class="card-subtext" style="font-size: 11px; margin-bottom: 10px;">Live PM2.5 & Spatial ML Model</div>

        <div class="clock-badge pill-bg">
            <span>🇰🇿 Almaty Time</span>
            <span id="almaty-clock" style="color:#3b82f6;">--:--:--</span>
        </div>

        <div class="mode-btn-group">
            <button class="mode-btn active" id="btn-auto" onclick="setThemeMode('auto')">Auto</button>
            <button class="mode-btn" id="btn-day" onclick="setThemeMode('day')">☀️ Day</button>
            <button class="mode-btn" id="btn-night" onclick="setThemeMode('night')">🌙 Night</button>
        </div>

        <div>
            <div style="display:flex; justify-content:space-between; font-size:10px;" class="card-subtext">
                <span>Bright (06:00)</span>
                <span id="slider-hour-label" style="font-weight:bold;">Auto Mode</span>
                <span>Dark (22:00)</span>
            </div>
            <input type="range" min="0" max="23" value="12" id="time-range-slider" class="time-slider" oninput="onSliderChange(this.value)">
        </div>

        <hr style="border:none; border-top: 1px solid rgba(150,150,150,0.2); margin: 10px 0;">

        <div style="font-size:11px; font-weight:700; margin-bottom:6px;">AQI PM2.5 Reference</div>
        <div class="legend-row"><span><i class="legend-dot" style="background:#00E400;"></i>Good</span><span class="card-subtext">0 - 12 µg/m³</span></div>
        <div class="legend-row"><span><i class="legend-dot" style="background:#FFFF00;"></i>Moderate</span><span class="card-subtext">12.1 - 35.4</span></div>
        <div class="legend-row"><span><i class="legend-dot" style="background:#FF7E00;"></i>Sensitive Groups</span><span class="card-subtext">35.5 - 55.4</span></div>
        <div class="legend-row"><span><i class="legend-dot" style="background:#FF0000;"></i>Unhealthy</span><span class="card-subtext">55.5 - 150.4</span></div>
        <div class="legend-row"><span><i class="legend-dot" style="background:#8F3F97;"></i>Very Unhealthy</span><span class="card-subtext">150.5 - 250.4</span></div>
        <div class="legend-row"><span><i class="legend-dot" style="background:#7E0023;"></i>Hazardous</span><span class="card-subtext">> 250.4</span></div>
    </div>

    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var map = null;
        for (var key in window) {{
            if (key.startsWith("map_") && window[key] instanceof L.Map) {{
                map = window[key];
                break;
            }}
        }}
        if (!map) return;

        // 100% Free Public Tiles (Zero CARTO Key Dependency)
        var esriStreet = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
            attribution: 'Esri, DeLorme, NAVTEQ',
            maxZoom: 18
        }});

        var esriDark = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
            attribution: 'Esri, HERE, Garmin, USGS',
            maxZoom: 16
        }});

        var osmFallback = L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 19
        }});

        // Auto Fallback Event Handler
        function attachFallback(layer) {{
            layer.on('tileerror', function() {{
                if (!map.hasLayer(osmFallback)) {{
                    map.removeLayer(layer);
                    osmFallback.addTo(map);
                }}
            }});
        }}
        attachFallback(esriStreet);
        attachFallback(esriDark);

        var activeTile = null;
        var currentMode = "auto";
        var knownStations = {js_known_data};

        function updateAlmatyClock() {{
            var now = new Date();
            var options = {{ timeZone: "Asia/Almaty", hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }};
            var dateOptions = {{ timeZone: "Asia/Almaty", year: 'numeric', month: 'short', day: 'numeric' }};
            
            var timeStr = new Intl.DateTimeFormat([], options).format(now);
            var dateStr = new Intl.DateTimeFormat([], dateOptions).format(now);

            var clockEl = document.getElementById("almaty-clock");
            if (clockEl) clockEl.innerText = timeStr + " (" + dateStr + ")";

            if (currentMode === "auto") {{
                var almatyHour = parseInt(new Intl.DateTimeFormat([], {{ timeZone: "Asia/Almaty", hour: 'numeric', hour12: false }}).format(now), 10);
                applyThemeForHour(almatyHour);
                document.getElementById("time-range-slider").value = almatyHour;
                document.getElementById("slider-hour-label").innerText = "Auto (" + almatyHour + ":00)";
            }}
        }}

        function applyThemeForHour(hour) {{
            var isDaytime = (hour >= 6 && hour < 18);
            setUITheme(isDaytime);
        }}

        function setUITheme(isDay) {{
            var body = document.body;
            var dayNightIcon = document.getElementById("day-night-icon");

            if (isDay) {{
                if (activeTile !== esriStreet) {{
                    if (activeTile) map.removeLayer(activeTile);
                    esriStreet.addTo(map);
                    activeTile = esriStreet;
                }}
                body.classList.remove("theme-dark");
                body.classList.add("theme-light");
                if (dayNightIcon) dayNightIcon.innerText = "☀️";
            }} else {{
                if (activeTile !== esriDark) {{
                    if (activeTile) map.removeLayer(activeTile);
                    esriDark.addTo(map);
                    activeTile = esriDark;
                }}
                body.classList.remove("theme-light");
                body.classList.add("theme-dark");
                if (dayNightIcon) dayNightIcon.innerText = "🌙";
            }}
        }}

        window.setThemeMode = function(mode) {{
            currentMode = mode;
            document.querySelectorAll(".mode-btn").forEach(btn => btn.classList.remove("active"));
            document.getElementById("btn-" + mode).classList.add("active");

            if (mode === "day") {{
                setUITheme(true);
                document.getElementById("slider-hour-label").innerText = "Manual Day";
            }} else if (mode === "night") {{
                setUITheme(false);
                document.getElementById("slider-hour-label").innerText = "Manual Night";
            }} else {{
                updateAlmatyClock();
            }}
        }};

        window.onSliderChange = function(hour) {{
            currentMode = "manual";
            document.querySelectorAll(".mode-btn").forEach(btn => btn.classList.remove("active"));
            document.getElementById("slider-hour-label").innerText = hour + ":00 Almaty";
            applyThemeForHour(parseInt(hour, 10));
        }};

        setInterval(updateAlmatyClock, 1000);
        updateAlmatyClock();

        // Inverse Distance Weighting (IDW) AI Interpolation
        function predictPM25(lat, lng) {{
            if (!knownStations || knownStations.length === 0) return 0.0;
            var num = 0.0, den = 0.0, p = 2.0;
            for (var i = 0; i < knownStations.length; i++) {{
                var st = knownStations[i];
                var d = Math.sqrt(Math.pow(lat - st.lat, 2) + Math.pow(lng - st.lon, 2));
                if (d < 0.0001) return st.pm25;
                var w = 1.0 / Math.pow(d, p);
                num += w * st.pm25;
                den += w;
            }}
            return den === 0 ? 0.0 : num / den;
        }}

        function getCategory(val) {{
            if (val <= 12.0) return {{ text: "Good", color: "#00E400" }};
            if (val <= 35.4) return {{ text: "Moderate", color: "#eab308" }};
            if (val <= 55.4) return {{ text: "Sensitive Groups", color: "#f97316" }};
            if (val <= 150.4) return {{ text: "Unhealthy", color: "#ef4444" }};
            if (val <= 250.4) return {{ text: "Very Unhealthy", color: "#8F3F97" }};
            return {{ text: "Hazardous", color: "#7E0023" }};
        }}

        var tooltipDiv = document.createElement("div");
        tooltipDiv.id = "ai-hover-card";
        tooltipDiv.style.position = "absolute";
        tooltipDiv.style.display = "none";
        tooltipDiv.style.zIndex = "10000";
        tooltipDiv.style.background = "rgba(15, 23, 42, 0.94)";
        tooltipDiv.style.color = "#ffffff";
        tooltipDiv.style.padding = "10px 14px";
        tooltipDiv.style.borderRadius = "10px";
        tooltipDiv.style.boxShadow = "0 8px 24px rgba(0,0,0,0.4)";
        tooltipDiv.style.fontFamily = "system-ui, -apple-system, sans-serif";
        tooltipDiv.style.fontSize = "12px";
        tooltipDiv.style.pointerEvents = "none";
        tooltipDiv.style.border = "1px solid rgba(255,255,255,0.15)";
        document.body.appendChild(tooltipDiv);

        map.on("mousemove", function(e) {{
            var lat = e.latlng.lat;
            var lng = e.latlng.lng;
            var predictedVal = predictPM25(lat, lng);
            var cat = getCategory(predictedVal);

            tooltipDiv.innerHTML = `
                <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px; font-weight:700; color:#a855f7;">
                    <span>🤖</span> <span>AI SPATIAL MODEL</span>
                </div>
                <div style="font-size:17px; font-weight:800; margin-bottom:2px;">
                    ${{predictedVal.toFixed(1)}} <span style="font-size:11px; font-weight:normal;">µg/m³</span>
                </div>
                <div style="display:inline-block; padding:2px 8px; border-radius:4px; font-weight:700; font-size:10px; background:${{cat.color}}; color:#fff; margin-bottom:4px;">
                    ${{cat.text}}
                </div>
                <div style="color:#94a3b8; font-size:10px;">
                    Lat: ${{lat.toFixed(4)}}, Lon: ${{lng.toFixed(4)}}
                </div>
            `;

            tooltipDiv.style.left = (e.originalEvent.pageX + 15) + "px";
            tooltipDiv.style.top = (e.originalEvent.pageY + 15) + "px";
            tooltipDiv.style.display = "block";
        }});

        map.on("mouseout", function() {{
            tooltipDiv.style.display = "none";
        }});
    }});
    </script>
    """

    m.get_root().html.add_child(folium.Element(custom_assets_html))
    m.save(output_html)
    print(f"🗺 Map generated successfully: {os.path.abspath(output_html)}")


def fetch_and_update():
    if not API_KEY:
        print("⚠ Warning: API_KEY environment variable is not set. Reading local CSV...")
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            render_folium_map(df)
        else:
            print("❌ Error: No local CSV found and API_KEY is missing.")
        return

    headers = {"x-api-key": API_KEY, "Accept": "application/json"}

    try:
        response = requests.get(API_URL, headers=headers, timeout=15)
        response.raise_for_status()

        records = extract_pm25_data(response.json())
        new_df = pd.DataFrame(records)

        if new_df.empty:
            print("⚠ API returned empty station list.")
            return

        if os.path.exists(CSV_FILE):
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
        print(f"✅ Dataset updated ({len(new_df)} station readings fetched).")

        render_folium_map(combined_df)

    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Failed: {e}")


if __name__ == "__main__":
    fetch_and_update()
