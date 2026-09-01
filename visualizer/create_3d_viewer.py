#!/usr/bin/env python3
import os
import sys
import json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw

# The DEM canvas is far above Pillow's default decompression-bomb limit.
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_source as ms

def main():
    print("=== DEM 3D Viewer Asset Generator ===")
    
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_path = os.path.join(project_root, "dem_generator", "dem_new_12k.png")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run the DEM generator first.")
        sys.exit(1)
        
    output_rgb_path = os.path.join(current_dir, "dem_1024_rgb16.png")
    output_texture_path = os.path.join(current_dir, "dem_1024_texture.png")
    output_html_path = os.path.join(current_dir, "dem_viewer_3d.html")
    
    # Target size for the web assets. The heightmap stays at 1024 (the mesh never asks
    # for more than 1024 vertices a side), but the colour texture is painted at twice
    # that: the hedgerows between fields are only ~8 m wide, which is barely one pixel
    # at 1024 over a 6 km canvas, and the field pattern washes out into one green blob.
    target_size = 1024

    # Real-world dimensions of the generated terrain. Taken from map_source so the
    # viewer cannot drift out of step with the DEM and OSM generators.
    dem_size_m = int(ms.CANVAS_M)
    playable_size_m = int(ms.PLAYABLE_M)

    # Colour texture resolution: aim for ~3 m per pixel so an 8 m hedgerow is still a
    # couple of pixels wide, capped at 4096 to keep the web asset a sane size.
    texture_size = 1 << max(11, min(12, int(round(np.log2(dem_size_m / 3.0)))))
    print(f"Texture resolution: {texture_size}x{texture_size} "
          f"({dem_size_m/texture_size:.1f} m per pixel)")
    
    # 1. Load and process heightmap
    print(f"Loading heightmap {input_path}...")
    img = Image.open(input_path)
    print(f"Original size: {img.size}, format: {img.format}, mode: {img.mode}")
    
    # Downsample to target size using bilinear interpolation
    print(f"Downsampling to {target_size}x{target_size}...")
    img_resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    data_resized = np.array(img_resized, dtype=np.float32)
    data_texture = np.array(img.resize((texture_size, texture_size),
                                       Image.Resampling.BILINEAR), dtype=np.float32)
    
    # Min/Max in raw values and meters
    h_min_raw = data_resized.min()
    h_max_raw = data_resized.max()
    h_min_m = h_min_raw / 100.0
    h_max_m = h_max_raw / 100.0
    print(f"Elevation range: {h_min_m:.2f}m to {h_max_m:.2f}m (raw: {h_min_raw:.1f} to {h_max_raw:.1f})")
    
    # 2. Save 16-bit RGB encoded heightmap
    # Red channel = height % 256
    # Green channel = height // 256
    # Blue channel = 0
    print("Encoding 16-bit heightmap to RGB PNG...")
    data_clipped = np.clip(data_resized, 0, 65535).astype(np.uint16)
    r = (data_clipped % 256).astype(np.uint8)
    g = ((data_clipped // 256) % 256).astype(np.uint8)
    b = np.zeros_like(r)
    
    rgb_data = np.dstack((r, g, b))
    img_rgb = Image.fromarray(rgb_data, mode="RGB")
    img_rgb.save(output_rgb_path)
    print(f"Saved RGB heightmap to: {output_rgb_path}")
    
    # 2.5. Parse OSM way coordinates to build polygon mask for texture coloring
    osm_candidates = [
        os.path.join(project_root, "osm_generator", "map.osm"),
        os.path.join(project_root, "osm_generator", "outputs", "zoning_map manual.osm"),
        os.path.join(project_root, "osm_generator", "outputs", "zoning_map.osm"),
        os.path.join(current_dir, "map.osm"),
    ]
    osm_path = next((p for p in osm_candidates if os.path.exists(p)), None)

    ways_data = []
    # Bounds of the OSM area; read from the file's <bounds> element (falls back to node extents).
    min_lat = min_lon = max_lat = max_lon = None
    if osm_path:
        print(f"Found OSM file at: {osm_path}. Parsing features...")
        try:
            tree = ET.parse(osm_path)
            root = tree.getroot()

            bounds = root.find("bounds")
            if bounds is not None:
                min_lat = float(bounds.get("minlat"))
                max_lat = float(bounds.get("maxlat"))
                min_lon = float(bounds.get("minlon"))
                max_lon = float(bounds.get("maxlon"))

            nodes = {}
            for node in root.findall("node"):
                nid = node.get("id")
                lat = float(node.get("lat"))
                lon = float(node.get("lon"))
                nodes[nid] = (lat, lon)

            for way in root.findall("way"):
                wid = way.get("id")
                tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
                refs = [nd.get("ref") for nd in way.findall("nd")]
                coords = [nodes[ref] for ref in refs if ref in nodes]

                if coords:
                    ways_data.append({
                        "id": wid,
                        "tags": tags,
                        "coords": coords
                    })
            print(f"Parsed {len(ways_data)} ways from OSM.")

            if min_lat is None and nodes:
                lats = [lat for lat, _ in nodes.values()]
                lons = [lon for _, lon in nodes.values()]
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)
                print("Warning: no <bounds> element, using node extents instead.")
        except Exception as e:
            print(f"Warning: Failed to parse OSM: {e}")
            ways_data = []
    else:
        print("No OSM file found. Skipping feature overlays.")

    if min_lat is None:
        # Neutral placeholder so the generated HTML stays valid without OSM data.
        min_lat, max_lat, min_lon, max_lon = 0.0, 1.0, 0.0, 1.0
    else:
        print(f"OSM bounds: lat {min_lat:.6f}..{max_lat:.6f}, lon {min_lon:.6f}..{max_lon:.6f}")

    osm_data_json = json.dumps(ways_data)
    
    # Generate the base color image (1024x1024)
    # Default background is a clean light gray
    bg_color = (204, 204, 204) # #CCCCCC
    base_color_img = Image.new("RGB", (texture_size, texture_size), bg_color)
    draw = ImageDraw.Draw(base_color_img)
    
    # Style rules matched against the way tags, in priority order.
    # (tag_key, tag_value or None for "any", hex color, line width in texture px)
    # natural=wood is checked before landuse: forests are tagged with both
    # natural=wood and landuse=farmyard, and the forest reading is the meaningful one.
    style_rules = [
        ("natural", "water", "#2563EB", 4),       # Water blue
        ("water", None, "#2563EB", 4),
        ("natural", "wood", "#166534", 4),        # Forest green
        ("landuse", "forest", "#166534", 4),
        ("landuse", "farmyard", "#EC4899", 4),    # Pink for farmyard
        ("landuse", "farmland", "#86EFAC", 3),    # Light green for farmland
        ("railway", None, "#F59E0B", 5),          # Amber railway
        ("highway", "primary", "#111827", 8),     # Road hierarchy, darkest = biggest
        ("highway", "secondary", "#374151", 5),
        ("highway", "tertiary", "#78716C", 3),
        ("highway", None, "#9CA3AF", 3),
    ]

    def hex_to_rgb(hex_str):
        h = hex_str.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def match_style(tags):
        for key, val, hex_color, width in style_rules:
            if key in tags and (val is None or tags[key] == val):
                return hex_to_rgb(hex_color), width
        return None, None

    # Convert lat/lon to pixel coordinates on the texture. The DEM canvas is wider than
    # the playable area, and only the playable band corresponds to the OSM data:
    #   band = texture_size * playable/canvas, centred, so the margins stay bare terrain.
    osm_band = texture_size * (playable_size_m / dem_size_m)
    osm_offset = (texture_size - osm_band) / 2

    def to_pixels(coords):
        pts = []
        for lat, lon in coords:
            u = (lon - min_lon) / (max_lon - min_lon)
            v = (max_lat - lat) / (max_lat - min_lat)
            pts.append((osm_offset + u * osm_band, osm_offset + v * osm_band))
        return pts

    if ways_data:
        polygons, lines = [], []
        for way in ways_data:
            tags = way.get("tags", {})
            coords = way.get("coords", [])
            if len(coords) < 2:
                continue

            color, width = match_style(tags)
            if color is None:
                continue

            pts = to_pixels(coords)
            is_closed = len(coords) > 2 and coords[0] == coords[-1]
            if is_closed:
                # Shoelace area, used to paint large areas first so small ones stay visible.
                area = abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1]
                               for i in range(len(pts)))) / 2.0
                polygons.append((area, pts, color))
            else:
                lines.append((pts, color, width))

        polygons.sort(key=lambda p: p[0], reverse=True)
        for _, pts, color in polygons:
            draw.polygon(pts, fill=color)
        for pts, color, width in lines:
            draw.line(pts, fill=color, width=width, joint="curve")

        print(f"Painted {len(polygons)} areas and {len(lines)} linear features onto the texture.")

    # Dashed outline of the playable area, so the border is readable in the texture.
    border_color = (255, 255, 255)
    b0, b1 = osm_offset, osm_offset + osm_band
    for start in range(int(b0), int(b1), 16):
        end = min(start + 8, b1)
        draw.line([(start, b0), (end, b0)], fill=border_color, width=2)
        draw.line([(start, b1), (end, b1)], fill=border_color, width=2)
        draw.line([(b0, start), (b0, end)], fill=border_color, width=2)
        draw.line([(b1, start), (b1, end)], fill=border_color, width=2)


    # 3. Bake the flat colors into a shaded relief texture
    def shade_and_save(color_img, path, label):
        """Applies hillshading from the DEM onto a flat color image and saves it."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib.colors import LightSource

            ls = LightSource(azdeg=315, altdeg=45)
            rgb_input = np.array(color_img, dtype=np.float32) / 255.0
            # The elevation grid has to match the image being shaded pixel for pixel.
            px_m = dem_size_m / texture_size
            shaded = ls.shade_rgb(rgb_input, elevation=data_texture / 100.0,
                                  blend_mode='overlay', vert_exag=1.5, dx=px_m, dy=px_m)
            Image.fromarray((shaded * 255).astype(np.uint8)).save(path)
            print(f"Saved shaded {label} to: {path}")
        except Exception as e:
            print(f"Warning: could not shade {label} with matplotlib: {e}")
            color_img.save(path)
            print(f"Saved flat (unshaded) {label} to: {path}")

    print("Generating shaded relief terrain texture...")
    shade_and_save(base_color_img, output_texture_path, "terrain texture")

    # 3.5. Soil type map from Precision Farming (pf_generator). A plain
    # soilMap.png wins; otherwise the newest of the seed-prefixed maps
    # (<field>_seed_<n>_soilMap.png) is used. The *_vis.png files are RGB
    # renderings, not index maps, so they never qualify.
    import glob
    soil_path = os.path.join(project_root, "pf_generator", "soilMap.png")
    if not os.path.exists(soil_path):
        seed_maps = [p for p in
                     glob.glob(os.path.join(project_root, "pf_generator", "*soilMap.png"))
                     if not p.endswith("_vis.png")]
        if seed_maps:
            soil_path = max(seed_maps, key=os.path.getmtime)
    output_soil_texture_path = os.path.join(current_dir, "soil_1024_texture.png")
    output_soil_index_path = os.path.join(current_dir, "soil_1024_index.png")

    # index -> (spanish name, english name, yield, color); mirrors pf_generator/generate_soil.py
    soil_types = [
        {"name": "Arena Limosa", "name_en": "Loamy Sand", "yield": "75%", "color": (220, 185, 80)},
        {"name": "Franco Arenoso", "name_en": "Sandy Loam", "yield": "100%", "color": (180, 130, 60)},
        {"name": "Franco", "name_en": "Loam", "yield": "125%", "color": (70, 150, 50)},
        {"name": "Arcilla Limosa", "name_en": "Silty Clay", "yield": "80%", "color": (120, 70, 160)},
    ]
    SOIL_NONE = 255  # marker for "outside the playable area"

    soil_available = False
    if os.path.exists(soil_path):
        print(f"Loading soil map {soil_path}...")
        soil_src = Image.open(soil_path)
        # P (indexed) and L images already hold the soil index per pixel; converting a
        # palette image to L would replace the indices with palette luminance.
        soil_idx_full = np.array(soil_src if soil_src.mode in ("P", "L") else soil_src.convert("L"))
        print(f"Soil map size: {soil_src.size}, mode: {soil_src.mode}, "
              f"types present: {sorted(np.unique(soil_idx_full).tolist())}")

        band_px = int(round(osm_band))
        offset_px = int(round(osm_offset))

        # Nearest neighbour keeps the soil indices intact (no blended in-between values)
        soil_band = np.array(Image.fromarray(soil_idx_full, mode="L").resize(
            (band_px, band_px), Image.Resampling.NEAREST))

        soil_index_map = np.full((texture_size, texture_size), SOIL_NONE, dtype=np.uint8)
        soil_index_map[offset_px:offset_px + band_px, offset_px:offset_px + band_px] = soil_band

        # Lookup image for the viewer: the soil index lives in the red channel.
        index_rgb = np.dstack((soil_index_map,
                               np.zeros_like(soil_index_map),
                               np.zeros_like(soil_index_map)))
        Image.fromarray(index_rgb, mode="RGB").save(output_soil_index_path)
        print(f"Saved soil index lookup to: {output_soil_index_path}")

        soil_color_img = np.full((texture_size, texture_size, 3), bg_color, dtype=np.uint8)
        for i, meta in enumerate(soil_types):
            soil_color_img[soil_index_map == i] = meta["color"]

        soil_pil = Image.fromarray(soil_color_img, mode="RGB")
        soil_draw = ImageDraw.Draw(soil_pil)
        for start in range(int(b0), int(b1), 16):
            end = min(start + 8, b1)
            soil_draw.line([(start, b0), (end, b0)], fill=border_color, width=2)
            soil_draw.line([(start, b1), (end, b1)], fill=border_color, width=2)
            soil_draw.line([(b0, start), (b0, end)], fill=border_color, width=2)
            soil_draw.line([(b1, start), (b1, end)], fill=border_color, width=2)

        print("Generating shaded soil type texture...")
        shade_and_save(soil_pil, output_soil_texture_path, "soil texture")
        soil_available = True
    else:
        print(f"No soil map at {soil_path}. The soil layer will be disabled in the viewer.")

    def rgb_to_hex(rgb):
        return "#{:02X}{:02X}{:02X}".format(*rgb)

    soil_types_json = json.dumps(
        [{"name": s["name"], "nameEn": s["name_en"], "yield": s["yield"],
          "color": rgb_to_hex(s["color"])} for s in soil_types],
        ensure_ascii=False)
    soil_available_js = "true" if soil_available else "false"
    soil_control_style = "" if soil_available else "display: none;"
    soil_legend_html = "\n".join(
        '                <div class="legend-item"><span class="legend-swatch" style="background:'
        + rgb_to_hex(s["color"]) + '"></span>' + s["name"] + " (" + s["yield"] + ")</div>"
        for s in soil_types)


    # 4. Generate HTML interactive 3D Viewer
    print("Writing HTML interactive 3D viewer...")
    
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualizador 3D - DEM Granja Bonita</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    
    <!-- Three.js and OrbitControls via CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <style>
        :root {{
            --bg-color: #0d0e12;
            --panel-bg: rgba(18, 20, 26, 0.75);
            --panel-border: rgba(255, 255, 255, 0.1);
            --accent-color: #4f46e5;
            --accent-hover: #6366f1;
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Outfit', sans-serif;
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }}

        #canvas-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        }}

        /* HUD overlay */
        .hud-panel {{
            position: absolute;
            z-index: 10;
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .hud-panel:hover {{
            border-color: rgba(255, 255, 255, 0.18);
        }}

        /* Header Panel */
        #header-panel {{
            top: 20px;
            left: 20px;
            max-width: 400px;
        }}

        h1 {{
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 4px;
            background: linear-gradient(135deg, #fff 30%, var(--text-muted) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            font-size: 13px;
            color: var(--text-muted);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 16px;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 10px;
            border: 1px solid rgba(255, 255, 255, 0.02);
        }}

        .stat-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }}

        .stat-value {{
            font-size: 16px;
            font-weight: 600;
        }}

        /* Control Panel */
        #control-panel {{
            top: 20px;
            right: 20px;
            width: 320px;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
        }}

        .control-group {{
            margin-bottom: 20px;
        }}

        .control-group:last-child {{
            margin-bottom: 0;
        }}

        .group-title {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .control-row {{
            margin-bottom: 12px;
        }}

        label {{
            display: block;
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}

        /* Inputs and Sliders */
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        input[type="range"] {{
            flex: 1;
            -webkit-appearance: none;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.1);
            outline: none;
        }}

        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-color);
            cursor: pointer;
            transition: background 0.2s;
        }}

        input[type="range"]::-webkit-slider-thumb:hover {{
            background: var(--accent-hover);
        }}

        .slider-value {{
            font-size: 12px;
            width: 35px;
            text-align: right;
            font-weight: 600;
        }}

        /* Buttons and Selectors */
        .btn-toggle-group {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
            background: rgba(0, 0, 0, 0.2);
            padding: 4px;
            border-radius: 8px;
        }}

        .btn-toggle {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 4px;
            border-radius: 6px;
            cursor: pointer;
            font-family: inherit;
            font-size: 11px;
            font-weight: 600;
            transition: all 0.2s;
        }}

        .btn-toggle.active {{
            background: var(--accent-color);
            color: #fff;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.4);
        }}

        .btn-toggle:hover:not(.active) {{
            color: var(--text-color);
            background: rgba(255, 255, 255, 0.05);
        }}

        .btn-primary {{
            background: var(--accent-color);
            border: none;
            color: #fff;
            padding: 10px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 13px;
            font-weight: 600;
            width: 100%;
            transition: all 0.2s;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }}

        .btn-primary:hover {{
            background: var(--accent-hover);
            box-shadow: 0 4px 16px rgba(79, 70, 229, 0.4);
        }}

        /* Switch checkbox */
        .switch-container {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }}

        .switch {{
            position: relative;
            display: inline-block;
            width: 40px;
            height: 20px;
        }}

        .switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}

        .switch-slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.1);
            transition: .3s;
            border-radius: 20px;
        }}

        .switch-slider:before {{
            position: absolute;
            content: "";
            height: 14px;
            width: 14px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }}

        input:checked + .switch-slider {{
            background-color: var(--accent-color);
        }}

        input:checked + .switch-slider:before {{
            transform: translateX(20px);
        }}

        /* OSM legend */
        .legend {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 6px 10px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            color: var(--text-muted);
        }}

        .legend-swatch {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            flex-shrink: 0;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }}

        /* First person mode */
        #crosshair {{
            position: absolute;
            top: 50%;
            left: 50%;
            width: 18px;
            height: 18px;
            margin: -9px 0 0 -9px;
            z-index: 12;
            display: none;
            pointer-events: none;
        }}

        #crosshair:before, #crosshair:after {{
            content: "";
            position: absolute;
            background: rgba(255, 255, 255, 0.75);
            box-shadow: 0 0 3px rgba(0, 0, 0, 0.8);
        }}

        #crosshair:before {{ left: 8px; top: 0; width: 2px; height: 18px; }}
        #crosshair:after {{ top: 8px; left: 0; height: 2px; width: 18px; }}

        #pov-panel {{
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: auto;
            max-width: 620px;
            display: none;
            padding: 14px 20px;
            text-align: center;
        }}

        #pov-panel .pov-keys {{
            display: flex;
            gap: 14px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 10px;
            font-size: 12px;
            color: var(--text-muted);
        }}

        #pov-panel .pov-keys span b {{
            color: var(--text-color);
            font-family: monospace;
            background: rgba(255, 255, 255, 0.08);
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 4px;
        }}

        /* Probe Panel */
        #probe-panel {{
            bottom: 20px;
            left: 20px;
            width: 320px;
            display: none;
        }}

        .probe-value {{
            font-family: monospace;
            font-size: 13px;
        }}

        /* Help overlay */
        #help-btn {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            color: var(--text-color);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 10;
            font-weight: bold;
            font-size: 18px;
            backdrop-filter: blur(12px);
        }}

        #help-modal {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) scale(0.9);
            z-index: 100;
            width: 90%;
            max-width: 450px;
            background: rgba(18, 20, 26, 0.95);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(20px);
            opacity: 0;
            pointer-events: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        #help-modal.active {{
            opacity: 1;
            pointer-events: auto;
            transform: translate(-50%, -50%) scale(1);
        }}

        .modal-title {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 16px;
        }}

        .modal-body p {{
            font-size: 14px;
            color: var(--text-muted);
            line-height: 1.6;
            margin-bottom: 16px;
        }}

        .modal-controls {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 12px;
            margin-top: 16px;
            font-size: 13px;
        }}

        .control-key {{
            background: rgba(255, 255, 255, 0.1);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: monospace;
            font-weight: bold;
            text-align: center;
        }}

        .close-modal {{
            margin-top: 24px;
        }}

        #loading-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--bg-color);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            transition: opacity 0.5s ease;
        }}

        .spinner {{
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top-color: var(--accent-color);
            animation: spin 1s ease-in-out infinite;
            margin-bottom: 20px;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        #loading-text {{
            font-size: 16px;
            font-weight: 600;
            letter-spacing: 1px;
            color: var(--text-color);
        }}

        #loading-progress {{
            font-size: 14px;
            color: var(--text-muted);
            margin-top: 8px;
        }}
    </style>
</head>
<body>

    <div id="loading-overlay">
        <div class="spinner"></div>
        <div id="loading-text">CARGANDO ELEVACIÓN 3D...</div>
        <div id="loading-progress">Inicializando WebGL</div>
    </div>

    <div id="canvas-container"></div>

    <!-- Header / Info Panel -->
    <div id="header-panel" class="hud-panel">
        <div class="subtitle">Farming Simulator 25</div>
        <h1>DEM Granja Bonita 3D</h1>
        <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">Procedural Heightmap &amp; Layout Viewer</p>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Lienzo DEM</div>
                <div class="stat-value">{dem_size_m/1000:.2f} × {dem_size_m/1000:.2f} km</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Rango Alturas</div>
                <div class="stat-value">{h_min_m:.1f}m - {h_max_m:.1f}m</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Área Jugable</div>
                <div class="stat-value">{playable_size_m/1000:.2f} × {playable_size_m/1000:.2f} km</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Vértices 3D</div>
                <div class="stat-value" id="mesh-vertices">262,144</div>
            </div>
        </div>
    </div>

    <!-- Controls Panel -->
    <div id="control-panel" class="hud-panel">
        <div class="control-group">
            <div class="group-title">Visualización</div>
            <div class="control-row">
                <label>Modo de Superficie</label>
                <div class="btn-toggle-group" id="mode-buttons">
                    <button class="btn-toggle active" onclick="setRenderMode('texture')">Textura</button>
                    <button class="btn-toggle" onclick="setRenderMode('elevation')">Elevación</button>
                    <button class="btn-toggle" onclick="setRenderMode('wireframe')">Malla</button>
                </div>
            </div>
            <div class="control-row">
                <label>Resolución de Malla (Vértices)</label>
                <div class="btn-toggle-group" id="res-buttons" style="grid-template-columns: repeat(3, 1fr);">
                    <button class="btn-toggle" onclick="changeMeshResolution(256)">256²</button>
                    <button class="btn-toggle active" onclick="changeMeshResolution(512)">512²</button>
                    <button class="btn-toggle" onclick="changeMeshResolution(1024)">1024²</button>
                </div>
            </div>
        </div>

        <div class="control-group">
            <div class="group-title">Parámetros del Relieve</div>
            <div class="control-row">
                <label>Exageración Vertical</label>
                <div class="slider-container">
                    <input type="range" id="exaggeration-slider" min="0.1" max="5.0" step="0.1" value="1.5" oninput="updateExaggeration(this.value)">
                    <div class="slider-value" id="exaggeration-val">1.5x</div>
                </div>
            </div>
        </div>

        <div class="control-group">
            <div class="group-title">Límites & Guías</div>
            
            <div class="switch-container">
                <span style="font-size: 13px;">Límite Jugable (8.19km)</span>
                <label class="switch">
                    <input type="checkbox" id="toggle-playable" checked onchange="togglePlayableBox(this.checked)">
                    <span class="switch-slider"></span>
                </label>
            </div>

            <div class="switch-container" style="{soil_control_style}">
                <span style="font-size: 13px;">Mapa de Suelo (Precision Farming)</span>
                <label class="switch">
                    <input type="checkbox" id="toggle-soil" onchange="toggleSoilLayer(this.checked)">
                    <span class="switch-slider"></span>
                </label>
            </div>

            <div class="legend" id="soil-legend" style="display: none;">
{soil_legend_html}
            </div>
        </div>

        <div class="control-group">
            <div class="group-title">Vista en Primera Persona</div>
            <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">
                Cámara a la altura de los ojos de un jugador (1.80 m) sobre el terreno.
            </p>
            <button class="btn-primary" id="pov-btn" onclick="togglePov(!povMode)">Entrar en Modo POV</button>
        </div>

        <div class="control-group">
            <div class="group-title">Iluminación (Sol)</div>
            <div class="control-row">
                <label>Dirección del Sol (Ángulo)</label>
                <div class="slider-container">
                    <input type="range" id="sun-angle" min="0" max="360" value="135" oninput="updateSunAngle(this.value)">
                    <div class="slider-value" id="sun-angle-val">135°</div>
                </div>
            </div>
            <div class="control-row">
                <label>Altitud del Sol</label>
                <div class="slider-container">
                    <input type="range" id="sun-alt" min="10" max="90" value="45" oninput="updateSunAltitude(this.value)">
                    <div class="slider-value" id="sun-alt-val">45°</div>
                </div>
            </div>
        </div>

        <button class="btn-primary" onclick="resetCamera()">Restablecer Cámara</button>
    </div>

    <!-- Hover Probe Panel -->
    <div id="probe-panel" class="hud-panel">
        <div class="group-title" style="margin-bottom: 8px;">Información de Punto</div>
        <div style="display: flex; flex-direction: column; gap: 6px;">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 12px; color: var(--text-muted);">Coordenadas X, Y:</span>
                <span class="probe-value" id="probe-coords">0m, 0m</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 12px; color: var(--text-muted);">Elevación Real:</span>
                <span class="probe-value" id="probe-height" style="color: #60a5fa; font-weight: bold;">0.0m</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 12px; color: var(--text-muted);">Zona:</span>
                <span class="probe-value" id="probe-zone" style="font-weight: 600;">Zona de Juego</span>
            </div>
            <div style="display: flex; justify-content: space-between;" id="probe-soil-row">
                <span style="font-size: 12px; color: var(--text-muted);">Tipo de Suelo:</span>
                <span class="probe-value" id="probe-soil" style="font-weight: 600;">-</span>
            </div>
        </div>
    </div>

    <div id="crosshair"></div>

    <div id="pov-panel" class="hud-panel">
        <div style="font-size: 13px; font-weight: 600;">
            Modo POV — altura de ojos <span id="pov-eye">1.80</span> m
        </div>
        <div class="pov-keys">
            <span><b>Clic</b>Capturar ratón</span>
            <span><b>W A S D</b>Moverse (25 m/s)</span>
            <span><b>Shift</b>Correr (120 m/s)</span>
            <span><b>Q / E</b>Altura de ojos</span>
            <span><b>Esc</b>Salir</span>
        </div>
    </div>

    <div id="help-btn" onclick="toggleHelp(true)">?</div>

    <div id="help-modal">
        <div class="modal-title">Navegación 3D</div>
        <div class="modal-body">
            <p>Usa tu ratón (o gestos táctiles) para rotar, trasladar y hacer zoom en el modelo del terreno:</p>
            <div class="modal-controls">
                <span class="control-key">Clic Izq + Arrastrar</span>
                <span>Rotar la cámara sobre el terreno</span>
                
                <span class="control-key">Rueda del Ratón</span>
                <span>Acercar y alejar (Zoom)</span>
                
                <span class="control-key">Clic Der + Arrastrar</span>
                <span>Trasladar / Panorámica (Mover la vista)</span>
            </div>
            <p style="margin-top: 16px;">Coloca el puntero del ratón sobre el mapa para analizar las coordenadas, la elevación y el tipo de suelo en tiempo real.</p>

            <div class="modal-title" style="font-size: 16px; margin-top: 20px;">Modo POV (primera persona)</div>
            <div class="modal-controls">
                <span class="control-key">P</span>
                <span>Entrar o salir de la vista de jugador (1.80 m)</span>

                <span class="control-key">W A S D</span>
                <span>Caminar (Shift para correr)</span>

                <span class="control-key">Ratón</span>
                <span>Mirar alrededor (haz clic para capturar el puntero)</span>

                <span class="control-key">Q / E</span>
                <span>Bajar o subir la altura de los ojos</span>
            </div>
            <p style="margin-top: 16px;">En modo POV la exageración vertical vuelve a 1x para que las pendientes se vean a escala real.</p>
        </div>
        <button class="btn-primary close-modal" onclick="toggleHelp(false)">¡Entendido!</button>
    </div>

    <script>
        // Elevation ranges from Python
        const MIN_HEIGHT = {h_min_m};
        const MAX_HEIGHT = {h_max_m};
        const MAP_SIZE = {dem_size_m};        // Full DEM canvas in metres
        const PLAYABLE_SIZE = {playable_size_m}; // Playable area, centered in the canvas
        const PLAYABLE_OFFSET = (MAP_SIZE - PLAYABLE_SIZE) / 2;

        // OSM bounds and data, read straight from the source .osm file
        const MIN_LON = {min_lon};
        const MAX_LON = {max_lon};
        const MIN_LAT = {min_lat};
        const MAX_LAT = {max_lat};
        const OSM_DATA = {osm_data_json};

        let container, scene, camera, renderer, controls;
        let terrainGeom, terrainMesh, terrainMaterial;
        let heightData = null; // Float32Array storing raw elevations in meters
        let heightWidth = 0, heightHeight = 0;
        
        let currentRes = 512;
        let renderMode = 'texture'; // 'texture', 'elevation', 'wireframe'
        let verticalExaggeration = 1.5;
        
        // Scene objects
        let sunLight, ambientLight;
        let playableBox;
        let skyDome;
        let zonePolygons = []; // World-space polygons used to name the zone under the cursor
        let raycaster, mouse;

        // Soil layer (Precision Farming)
        const SOIL_AVAILABLE = {soil_available_js};
        const SOIL_TYPES = {soil_types_json};
        const SOIL_NONE = 255;
        let soilTexture = null;
        let soilData = null;      // Uint8Array of soil indices, one per texture pixel
        let soilWidth = 0, soilHeight = 0;
        let soilLayerOn = false;

        // First person (POV) mode
        let povMode = false;
        let povYaw = 0, povPitch = 0;
        let povEyeHeight = 1.8;   // metres above the ground
        // Deliberately far above human speed: the map is 12km wide and waiting is worse
        // than losing realism.
        const POV_WALK_SPEED = 25.0;   // m/s (~90 km/h)
        const POV_RUN_SPEED = 120.0;   // m/s, crosses the playable area in ~70s
        const povKeys = {{}};
        let povSavedState = null;
        let lastFrameTime = 0;

        // Feature types used to name what is under the cursor (read from the source .osm).
        // Forests carry both natural=wood and landuse=farmyard, so wood is matched first.
        const ZONE_TYPES = [
            {{ key: 'natural',  val: 'water',     color: '#60A5FA', label: 'Agua' }},
            {{ key: 'water',    val: null,        color: '#60A5FA', label: 'Agua' }},
            {{ key: 'natural',  val: 'wood',      color: '#22C55E', label: 'Bosque' }},
            {{ key: 'landuse',  val: 'forest',    color: '#22C55E', label: 'Bosque' }},
            {{ key: 'landuse',  val: 'farmyard',  color: '#EC4899', label: 'Farmyard' }},
            {{ key: 'landuse',  val: 'farmland',  color: '#86EFAC', label: 'Farmland' }}
        ];

        function matchZoneType(tags) {{
            for (const rule of ZONE_TYPES) {{
                if (rule.key in tags && (rule.val === null || tags[rule.key] === rule.val)) {{
                    return rule;
                }}
            }}
            return null;
        }}

        // OSM lat/lon -> world metres. The OSM area covers only the central playable band.
        function osmToWorld(lat, lon) {{
            const u = (lon - MIN_LON) / (MAX_LON - MIN_LON);
            const v = (MAX_LAT - lat) / (MAX_LAT - MIN_LAT);
            return [(u - 0.5) * PLAYABLE_SIZE, (v - 0.5) * PLAYABLE_SIZE];
        }}

        // World metres -> elevation in metres (unexaggerated)
        function heightAtWorld(x, z) {{
            const u = (x + MAP_SIZE / 2) / MAP_SIZE;
            const v = (z + MAP_SIZE / 2) / MAP_SIZE;
            if (u < 0 || u > 1 || v < 0 || v > 1) return 0;
            return getInterpolatedHeight(u, v);
        }}
        
        // Textures
        let colorTexture, heightmapImage;

        // Initialize App
        window.onload = function() {{
            init();
        }};

        function init() {{
            container = document.getElementById('canvas-container');
            
            // Set up Scene, Camera, Renderer
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0d0e12);
            scene.fog = new THREE.FogExp2(0x0d0e12, 0.0001);

            camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 10, 30000);
            
            // Logarithmic depth keeps the 0.3m near plane of the POV camera usable
            // together with the 30km far plane needed for the full canvas.
            renderer = new THREE.WebGLRenderer({{
                antialias: true,
                alpha: false,
                logarithmicDepthBuffer: true
            }});
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.appendChild(renderer.domElement);

            // Orbit Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = false;
            controls.maxPolarAngle = Math.PI / 2 - 0.05; // Don't go below ground
            controls.minDistance = 50;
            controls.maxDistance = 25000;
            
            // Default camera view
            resetCamera();

            // Lighting
            ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
            scene.add(ambientLight);

            sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
            sunLight.castShadow = true;
            sunLight.shadow.mapSize.width = 2048;
            sunLight.shadow.mapSize.height = 2048;
            sunLight.shadow.camera.near = 100;
            sunLight.shadow.camera.far = 20000;
            const d = 7000;
            sunLight.shadow.camera.left = -d;
            sunLight.shadow.camera.right = d;
            sunLight.shadow.camera.top = d;
            sunLight.shadow.camera.bottom = -d;
            scene.add(sunLight);
            
            updateSunPosition(135, 45);

            // Setup Raycasting
            raycaster = new THREE.Raycaster();
            mouse = new THREE.Vector2();

            // Sky used by the first person view
            buildSky();

            // Name lookup for the areas of the map
            buildZoneIndex();

            if (!SOIL_AVAILABLE) {{
                document.getElementById('probe-soil-row').style.display = 'none';
            }}

            // Load assets
            loadAssets();

            // Resize handler
            window.addEventListener('resize', onWindowResize, false);

            // Mouse move for terrain inspection
            window.addEventListener('mousemove', onMouseMove, false);

            // First person input
            window.addEventListener('keydown', onKeyDown, false);
            window.addEventListener('keyup', onKeyUp, false);
            renderer.domElement.addEventListener('click', function() {{
                if (povMode && document.pointerLockElement !== renderer.domElement) {{
                    renderer.domElement.requestPointerLock();
                }}
            }}, false);

            // Start Loop
            lastFrameTime = performance.now();
            animate();
        }}

        function resetCamera() {{
            // Framed for the full 12.29km canvas
            camera.position.set(0, 6500, 9500);
            controls.target.set(0, 100, 0);
            controls.update();
        }}

        function updateSunPosition(angleDeg, altitudeDeg) {{
            const angleRad = (angleDeg * Math.PI) / 180;
            const altRad = (altitudeDeg * Math.PI) / 180;
            
            const r = 8000;
            const y = r * Math.sin(altRad);
            const x = r * Math.cos(altRad) * Math.cos(angleRad);
            const z = r * Math.cos(altRad) * Math.sin(angleRad);
            
            sunLight.position.set(x, y, z);
        }}

        function loadAssets() {{
            const loadingProgress = document.getElementById('loading-progress');
            
            // Load visual texture & RGB Heightmap
            const textureLoader = new THREE.TextureLoader();
            
            loadingProgress.innerText = "Cargando Textura del Mapa...";
            
            // Cache-busting query parameter to force reloading the images from disk
            const cb = '?t=' + Date.now();
            
            loadSoilLayer();

            textureLoader.load('dem_1024_texture.png' + cb, function(tex) {{
                colorTexture = tex;
                colorTexture.wrapS = THREE.ClampToEdgeWrapping;
                colorTexture.wrapT = THREE.ClampToEdgeWrapping;
                colorTexture.anisotropy = renderer.capabilities.getMaxAnisotropy();

                loadingProgress.innerText = "Cargando Datos de Elevación (16-bit)...";
                
                const img = new Image();
                img.src = 'dem_1024_rgb16.png' + cb;
                img.onload = function() {{
                    heightmapImage = img;
                    
                    // Parse RGB image to heights
                    parseHeightmap(img);
                    
                    // Build terrain mesh
                    buildTerrainMesh(currentRes);
                    
                    // Build auxiliary lines/guides
                    buildGuides();

                    // Remove loading overlay
                    const loader = document.getElementById('loading-overlay');
                    loader.style.opacity = 0;
                    setTimeout(() => loader.style.display = 'none', 500);
                }};
            }}, undefined, function(err) {{
                console.error("Error loading terrain texture", err);
                loadingProgress.innerText = "Error al cargar texturas. Iniciando con colores planos...";
                // Fallback heightmap parse
                const img = new Image();
                img.src = 'dem_1024_rgb16.png' + cb;
                img.onload = function() {{
                    heightmapImage = img;
                    parseHeightmap(img);
                    buildTerrainMesh(currentRes);
                    buildGuides();
                    document.getElementById('loading-overlay').style.display = 'none';
                }};
            }});
        }}

        function parseHeightmap(img) {{
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            
            const imgData = ctx.getImageData(0, 0, img.width, img.height);
            const data = imgData.data;
            
            heightWidth = img.width;
            heightHeight = img.height;
            heightData = new Float32Array(heightWidth * heightHeight);
            
            for (let i = 0; i < heightData.length; i++) {{
                const r = data[i * 4];
                const g = data[i * 4 + 1];
                // Decode 16-bit value (in cm) and convert to meters
                const rawHeight = r + g * 256;
                heightData[i] = rawHeight / 100.0;
            }}
        }}

        function getInterpolatedHeight(u, v) {{
            if (!heightData) return 0;
            
            // Map u, v (0 to 1) to image pixel coords
            const px = u * (heightWidth - 1);
            const py = v * (heightHeight - 1);
            
            const x0 = Math.floor(px);
            const y0 = Math.floor(py);
            const x1 = Math.min(x0 + 1, heightWidth - 1);
            const y1 = Math.min(y0 + 1, heightHeight - 1);
            
            const tx = px - x0;
            const ty = py - y0;
            
            const h00 = heightData[y0 * heightWidth + x0];
            const h10 = heightData[y0 * heightWidth + x1];
            const h01 = heightData[y1 * heightWidth + x0];
            const h11 = heightData[y1 * heightWidth + x1];
            
            // Bilinear interpolation
            const h0 = h00 * (1 - tx) + h10 * tx;
            const h1 = h01 * (1 - tx) + h11 * tx;
            return h0 * (1 - ty) + h1 * ty;
        }}

        function buildTerrainMesh(res) {{
            if (terrainMesh) {{
                scene.remove(terrainMesh);
                terrainGeom.dispose();
            }}
            
            document.getElementById('mesh-vertices').innerText = (res * res).toLocaleString();
            
            // Geometry spans the whole DEM canvas; the playable area is the centre of it
            terrainGeom = new THREE.PlaneGeometry(MAP_SIZE, MAP_SIZE, res - 1, res - 1);
            
            // Displace plane vertices along Y (originally Z before rotation)
            const posAttr = terrainGeom.attributes.position;
            const count = posAttr.count;
            
            for (let i = 0; i < count; i++) {{
                // PlaneCoordinates are from -MAP_SIZE/2 to MAP_SIZE/2
                const x = posAttr.getX(i);
                const z = posAttr.getY(i);
                
                // Map x,z (-4096 to 4096) to u,v (0 to 1)
                const u = (x + MAP_SIZE / 2) / MAP_SIZE;
                const v = 1 - (z + MAP_SIZE / 2) / MAP_SIZE;
                
                const height = getInterpolatedHeight(u, v);
                posAttr.setZ(i, height * verticalExaggeration);
            }}
            
            // Rotate the plane to sit flat horizontally
            terrainGeom.rotateX(-Math.PI / 2);
            terrainGeom.computeVertexNormals();
            
            // Materials
            if (renderMode === 'texture') {{
                terrainMaterial = new THREE.MeshStandardMaterial({{
                    map: (soilLayerOn && soilTexture) ? soilTexture : colorTexture,
                    roughness: 0.85,
                    metalness: 0.1,
                    flatShading: false
                }});
            }} else if (renderMode === 'elevation') {{
                buildVertexColors(res);
                terrainMaterial = new THREE.MeshStandardMaterial({{
                    vertexColors: true,
                    roughness: 0.8,
                    metalness: 0.1
                }});
            }} else {{
                // Wireframe
                terrainMaterial = new THREE.MeshBasicMaterial({{
                    color: 0x6366f1,
                    wireframe: true
                }});
            }}
            
            terrainMesh = new THREE.Mesh(terrainGeom, terrainMaterial);
            terrainMesh.receiveShadow = true;
            terrainMesh.castShadow = true;
            scene.add(terrainMesh);
        }}

        function buildVertexColors(res) {{
            const count = terrainGeom.attributes.position.count;
            const colors = [];
            
            // Color palettes representing elevations
            // Gradient from Green (lowlands) -> Yellow -> Brown -> White (peaks)
            const colorRamp = [
                {{ h: MIN_HEIGHT, c: new THREE.Color(0x1e4620) }},
                {{ h: MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * 0.05, c: new THREE.Color(0x2d6a2e) }},
                {{ h: MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * 0.15, c: new THREE.Color(0x658c43) }},
                {{ h: MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * 0.40, c: new THREE.Color(0xb8a174) }},
                {{ h: MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * 0.60, c: new THREE.Color(0x8e7355) }},
                {{ h: MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * 0.80, c: new THREE.Color(0x5c5247) }},
                {{ h: MAX_HEIGHT, c: new THREE.Color(0xffffff) }}
            ];
            
            const posAttr = terrainGeom.attributes.position;
            for (let i = 0; i < count; i++) {{
                const yVal = posAttr.getY(i) / verticalExaggeration;
                
                let col = new THREE.Color(0xffffff);
                if (yVal <= colorRamp[0].h) {{
                    col.copy(colorRamp[0].c);
                }} else if (yVal >= colorRamp[colorRamp.length - 1].h) {{
                    col.copy(colorRamp[colorRamp.length - 1].c);
                }} else {{
                    for (let j = 0; j < colorRamp.length - 1; j++) {{
                        const lower = colorRamp[j];
                        const upper = colorRamp[j+1];
                        if (yVal >= lower.h && yVal <= upper.h) {{
                            const t = (yVal - lower.h) / (upper.h - lower.h);
                            col.copy(lower.c).lerp(upper.c, t);
                            break;
                        }}
                    }}
                }}
                colors.push(col.r, col.g, col.b);
            }}
            
            terrainGeom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        }}

        function buildGuides() {{
            // 1. Playable Area outline box (4km x 4km centered, height matches elevation bounds)
            const playMin = -PLAYABLE_SIZE / 2;
            const playMax = PLAYABLE_SIZE / 2;
            
            const boxGeom = new THREE.BufferGeometry();
            const yMin = MIN_HEIGHT * verticalExaggeration;
            const yMax = MAX_HEIGHT * verticalExaggeration;
            
            const vertices = [
                // Floor
                playMin, yMin, playMin,  playMax, yMin, playMin,
                playMax, yMin, playMin,  playMax, yMin, playMax,
                playMax, yMin, playMax,  playMin, yMin, playMax,
                playMin, yMin, playMax,  playMin, yMin, playMin,
                // Ceiling
                playMin, yMax, playMin,  playMax, yMax, playMin,
                playMax, yMax, playMin,  playMax, yMax, playMax,
                playMax, yMax, playMax,  playMin, yMax, playMax,
                playMin, yMax, playMax,  playMin, yMax, playMin,
                // Pillars
                playMin, yMin, playMin,  playMin, yMax, playMin,
                playMax, yMin, playMin,  playMax, yMax, playMin,
                playMax, yMin, playMax,  playMax, yMax, playMax,
                playMin, yMin, playMax,  playMin, yMax, playMax,
            ];
            boxGeom.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            
            const boxMat = new THREE.LineBasicMaterial({{ 
                color: 0x4f46e5, 
                linewidth: 2, 
                transparent: true,
                opacity: 0.8
            }});
            playableBox = new THREE.LineSegments(boxGeom, boxMat);
            playableBox.visible = !povMode && document.getElementById('toggle-playable').checked;
            scene.add(playableBox);
        }}

        // Builds the lookup used to name the area under the cursor. No 3D objects involved.
        function buildZoneIndex() {{
            zonePolygons = [];

            for (const way of OSM_DATA) {{
                const tags = way.tags || {{}};
                const coords = way.coords || [];
                if (coords.length < 3) continue;

                const isClosed =
                    coords[0][0] === coords[coords.length - 1][0] &&
                    coords[0][1] === coords[coords.length - 1][1];
                if (!isClosed) continue;

                const type = matchZoneType(tags);
                if (!type) continue;

                const flat = coords.map(([lat, lon]) => osmToWorld(lat, lon));
                zonePolygons.push({{
                    points: flat,
                    label: tags.name ? `${{type.label}} · ${{tags.name}}` : type.label,
                    color: type.color,
                    area: polygonArea(flat)
                }});
            }}

            // Smallest area first, so an inner feature wins over the field enclosing it.
            zonePolygons.sort((a, b) => a.area - b.area);
        }}

        function polygonArea(pts) {{
            let sum = 0;
            for (let i = 0; i < pts.length; i++) {{
                const [x0, z0] = pts[i];
                const [x1, z1] = pts[(i + 1) % pts.length];
                sum += x0 * z1 - x1 * z0;
            }}
            return Math.abs(sum) / 2;
        }}

        function pointInPolygon(x, z, pts) {{
            let inside = false;
            for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {{
                const [xi, zi] = pts[i];
                const [xj, zj] = pts[j];
                if ((zi > z) !== (zj > z) &&
                    x < (xj - xi) * (z - zi) / (zj - zi) + xi) {{
                    inside = !inside;
                }}
            }}
            return inside;
        }}

        function findZone(x, z) {{
            for (const poly of zonePolygons) {{
                if (pointInPolygon(x, z, poly.points)) return poly;
            }}
            return null;
        }}

        // --- Soil layer (Precision Farming) -------------------------------------

        function soilAt(x, z) {{
            if (!soilData) return null;

            const u = (x + MAP_SIZE / 2) / MAP_SIZE;
            const v = (z + MAP_SIZE / 2) / MAP_SIZE;
            if (u < 0 || u > 1 || v < 0 || v > 1) return null;

            const px = Math.min(soilWidth - 1, Math.floor(u * soilWidth));
            const py = Math.min(soilHeight - 1, Math.floor(v * soilHeight));
            const idx = soilData[py * soilWidth + px];

            return (idx === SOIL_NONE || idx >= SOIL_TYPES.length) ? null : SOIL_TYPES[idx];
        }}

        function loadSoilLayer() {{
            if (!SOIL_AVAILABLE) return;

            const cb = '?t=' + Date.now();

            new THREE.TextureLoader().load('soil_1024_texture.png' + cb, function(tex) {{
                soilTexture = tex;
                soilTexture.wrapS = THREE.ClampToEdgeWrapping;
                soilTexture.wrapT = THREE.ClampToEdgeWrapping;
                soilTexture.anisotropy = renderer.capabilities.getMaxAnisotropy();
                if (soilLayerOn) applySurfaceTexture();
            }});

            // Index image: soil type per pixel is stored in the red channel
            const img = new Image();
            img.src = 'soil_1024_index.png' + cb;
            img.onload = function() {{
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);

                const data = ctx.getImageData(0, 0, img.width, img.height).data;
                soilWidth = img.width;
                soilHeight = img.height;
                soilData = new Uint8Array(soilWidth * soilHeight);
                for (let i = 0; i < soilData.length; i++) {{
                    soilData[i] = data[i * 4];
                }}
            }};
        }}

        function applySurfaceTexture() {{
            if (!terrainMaterial || renderMode !== 'texture') return;
            const tex = (soilLayerOn && soilTexture) ? soilTexture : colorTexture;
            if (tex) {{
                terrainMaterial.map = tex;
                terrainMaterial.needsUpdate = true;
            }}
        }}

        function toggleSoilLayer(on) {{
            soilLayerOn = on;
            document.getElementById('soil-legend').style.display = on ? 'grid' : 'none';

            // The soil map only makes sense on the textured surface
            if (on && renderMode !== 'texture') setRenderMode('texture');
            applySurfaceTexture();
        }}

        // --- First person view (player eye height) -------------------------------

        function buildSky() {{
            // Vertical gradient painted on a canvas and wrapped on a large inverted sphere,
            // so the POV camera has a horizon to look at instead of the void.
            const c = document.createElement('canvas');
            c.width = 4;
            c.height = 256;
            const ctx = c.getContext('2d');
            const grad = ctx.createLinearGradient(0, 0, 0, 256);
            grad.addColorStop(0.00, '#1e3f73');
            grad.addColorStop(0.45, '#6f9bc9');
            grad.addColorStop(0.82, '#c4d2e0');
            grad.addColorStop(1.00, '#e8d9bb');
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, 4, 256);

            const geom = new THREE.SphereGeometry(20000, 32, 20);
            const mat = new THREE.MeshBasicMaterial({{
                map: new THREE.CanvasTexture(c),
                side: THREE.BackSide,
                fog: false,
                depthWrite: false
            }});

            skyDome = new THREE.Mesh(geom, mat);
            skyDome.visible = false;
            scene.add(skyDome);
        }}

        function togglePov(enable) {{
            if (enable === povMode) return;
            if (enable && !heightData) return; // terrain not loaded yet

            povMode = enable;

            const povPanel = document.getElementById('pov-panel');
            const crosshair = document.getElementById('crosshair');
            const btn = document.getElementById('pov-btn');

            if (enable) {{
                povSavedState = {{
                    position: camera.position.clone(),
                    target: controls.target.clone(),
                    exaggeration: verticalExaggeration,
                    near: camera.near,
                    fov: camera.fov,
                    fogColor: scene.fog.color.getHex()
                }};

                // At eye level only the real, unexaggerated elevations make sense
                if (verticalExaggeration !== 1.0) {{
                    document.getElementById('exaggeration-slider').value = 1.0;
                    updateExaggeration(1.0);
                }}

                controls.enabled = false;
                camera.near = 0.3;
                camera.fov = 70;
                camera.updateProjectionMatrix();
                camera.rotation.order = 'YXZ';

                // Drop the player where the orbit camera was looking, facing the same way
                const dir = new THREE.Vector3().subVectors(povSavedState.target, povSavedState.position);
                povYaw = Math.atan2(-dir.x, -dir.z);
                povPitch = 0;

                const limit = MAP_SIZE / 2 - 10;
                const x = Math.max(-limit, Math.min(limit, povSavedState.target.x));
                const z = Math.max(-limit, Math.min(limit, povSavedState.target.z));
                camera.position.set(x, heightAtWorld(x, z) + povEyeHeight, z);

                scene.fog.color.setHex(0xbfc9d4); // daytime haze instead of the dark void
                skyDome.visible = true;
                // The guide box would hang across the sky at eye level
                if (playableBox) playableBox.visible = false;

                povPanel.style.display = 'block';
                crosshair.style.display = 'block';
                btn.innerText = 'Salir del Modo POV';
                btn.blur(); // keep the keyboard on the camera, not on the button
                renderer.domElement.requestPointerLock();
            }} else {{
                if (document.pointerLockElement) document.exitPointerLock();

                controls.enabled = true;
                camera.near = povSavedState.near;
                camera.fov = povSavedState.fov;
                camera.updateProjectionMatrix();
                camera.rotation.set(0, 0, 0);
                camera.position.copy(povSavedState.position);
                controls.target.copy(povSavedState.target);
                controls.update();

                scene.fog.color.setHex(povSavedState.fogColor);
                skyDome.visible = false;
                if (playableBox) {{
                    playableBox.visible = document.getElementById('toggle-playable').checked;
                }}

                povPanel.style.display = 'none';
                crosshair.style.display = 'none';
                btn.innerText = 'Entrar en Modo POV';

                if (povSavedState.exaggeration !== verticalExaggeration) {{
                    document.getElementById('exaggeration-slider').value = povSavedState.exaggeration;
                    updateExaggeration(povSavedState.exaggeration);
                }}

                for (const k in povKeys) povKeys[k] = false;
            }}
        }}

        function updatePov(dt) {{
            camera.rotation.y = povYaw;
            camera.rotation.x = povPitch;
            camera.rotation.z = 0;

            const running = povKeys['ShiftLeft'] || povKeys['ShiftRight'];
            const speed = running ? POV_RUN_SPEED : POV_WALK_SPEED;

            let fwd = 0, side = 0;
            if (povKeys['KeyW'] || povKeys['ArrowUp']) fwd += 1;
            if (povKeys['KeyS'] || povKeys['ArrowDown']) fwd -= 1;
            if (povKeys['KeyD'] || povKeys['ArrowRight']) side += 1;
            if (povKeys['KeyA'] || povKeys['ArrowLeft']) side -= 1;

            if (fwd !== 0 || side !== 0) {{
                const len = Math.hypot(fwd, side);
                fwd /= len;
                side /= len;

                const sinY = Math.sin(povYaw), cosY = Math.cos(povYaw);
                const dx = (-sinY * fwd + cosY * side) * speed * dt;
                const dz = (-cosY * fwd - sinY * side) * speed * dt;

                const limit = MAP_SIZE / 2 - 10;
                camera.position.x = Math.max(-limit, Math.min(limit, camera.position.x + dx));
                camera.position.z = Math.max(-limit, Math.min(limit, camera.position.z + dz));
            }}

            // Stick to the ground
            const ground = heightAtWorld(camera.position.x, camera.position.z) * verticalExaggeration;
            camera.position.y = ground + povEyeHeight;
            skyDome.position.set(camera.position.x, 0, camera.position.z);
        }}

        function onKeyDown(e) {{
            if (e.code === 'Escape' && povMode) {{
                togglePov(false);
                return;
            }}
            if (e.code === 'KeyP' && !e.repeat) {{
                togglePov(!povMode);
                return;
            }}
            if (!povMode) return;

            povKeys[e.code] = true;

            if (e.code === 'KeyQ' || e.code === 'KeyE') {{
                povEyeHeight = Math.max(0.4, Math.min(8.0,
                    povEyeHeight + (e.code === 'KeyE' ? 0.2 : -0.2)));
                document.getElementById('pov-eye').innerText = povEyeHeight.toFixed(2);
            }}

            if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space',
                 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].indexOf(e.code) !== -1) {{
                e.preventDefault();
            }}
        }}

        function onKeyUp(e) {{
            povKeys[e.code] = false;
        }}

        function updateExaggeration(val) {{
            verticalExaggeration = parseFloat(val);
            document.getElementById('exaggeration-val').innerText = val + 'x';
            
            if (heightData) {{
                buildTerrainMesh(currentRes);

                // Get the current visible state of playableBox before recreating it
                const wasPlayableVisible = playableBox ? playableBox.visible : false;

                if (playableBox) {{
                    scene.remove(playableBox);
                }}

                buildGuides();

                // Restore the visibility state
                if (playableBox) {{
                    playableBox.visible = wasPlayableVisible;
                }}
            }}
        }}

        function setRenderMode(mode) {{
            renderMode = mode;
            
            const buttons = document.querySelectorAll('#mode-buttons .btn-toggle');
            buttons.forEach(btn => btn.classList.remove('active'));

            if (mode === 'texture') buttons[0].classList.add('active');
            if (mode === 'elevation') buttons[1].classList.add('active');
            if (mode === 'wireframe') buttons[2].classList.add('active');
            
            if (heightData) {{
                buildTerrainMesh(currentRes);
            }}
        }}

        function changeMeshResolution(res) {{
            currentRes = res;
            
            const buttons = document.querySelectorAll('#res-buttons .btn-toggle');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            if (res === 256) buttons[0].classList.add('active');
            if (res === 512) buttons[1].classList.add('active');
            if (res === 1024) buttons[2].classList.add('active');
            
            if (heightData) {{
                const overlay = document.getElementById('loading-overlay');
                const pText = document.getElementById('loading-progress');
                document.getElementById('loading-text').innerText = "RECONSTRUYENDO MALLA 3D...";
                pText.innerText = "Calculando vértices a " + res + "²...";
                overlay.style.display = 'flex';
                overlay.style.opacity = 1;
                
                setTimeout(() => {{
                    buildTerrainMesh(res);
                    overlay.style.opacity = 0;
                    setTimeout(() => overlay.style.display = 'none', 300);
                }}, 50);
            }}
        }}

        function togglePlayableBox(visible) {{
            if (playableBox) playableBox.visible = visible;
        }}



        function updateSunAngle(angle) {{
            document.getElementById('sun-angle-val').innerText = angle + '°';
            const alt = document.getElementById('sun-alt').value;
            updateSunPosition(parseInt(angle), parseInt(alt));
        }}

        function updateSunAltitude(alt) {{
            document.getElementById('sun-alt-val').innerText = alt + '°';
            const angle = document.getElementById('sun-angle').value;
            updateSunPosition(parseInt(angle), parseInt(alt));
        }}

        function toggleHelp(show) {{
            const modal = document.getElementById('help-modal');
            if (show) {{
                modal.classList.add('active');
            }} else {{
                modal.classList.remove('active');
            }}
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function onMouseMove(event) {{
            if (povMode) {{
                if (document.pointerLockElement === renderer.domElement) {{
                    povYaw -= (event.movementX || 0) * 0.0022;
                    povPitch -= (event.movementY || 0) * 0.0022;
                    const maxPitch = Math.PI / 2 - 0.05;
                    povPitch = Math.max(-maxPitch, Math.min(maxPitch, povPitch));
                }}
                // The probe reads whatever the crosshair is pointing at
                mouse.set(0, 0);
                return;
            }}

            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        }}

        function checkTerrainIntersection() {{
            if (!terrainMesh || !heightData) return;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(terrainMesh);
            
            const probePanel = document.getElementById('probe-panel');
            
            if (intersects.length > 0) {{
                const point = intersects[0].point;
                const x = point.x;
                const z = point.z;
                
                if (Math.abs(x) <= MAP_SIZE/2 && Math.abs(z) <= MAP_SIZE/2) {{
                    probePanel.style.display = 'block';
                    const realY = point.y / verticalExaggeration;
                    
                    // Local coordinates inside the playable area (0..PLAYABLE_SIZE, origin NW),
                    // the coordinate system used by the OSM and DEM generators.
                    const localX = x + PLAYABLE_SIZE / 2;
                    const localY = z + PLAYABLE_SIZE / 2;

                    const inPlayable = Math.abs(x) <= PLAYABLE_SIZE/2 && Math.abs(z) <= PLAYABLE_SIZE/2;

                    document.getElementById('probe-coords').innerText = inPlayable
                        ? `X: ${{Math.round(localX)}}m | Y: ${{Math.round(localY)}}m`
                        : `X: ${{Math.round(x)}}m | Z: ${{Math.round(z)}}m (global)`;
                    document.getElementById('probe-height').innerText = `${{realY.toFixed(2)}}m`;

                    const zoneLabel = document.getElementById('probe-zone');

                    if (inPlayable) {{
                        const zone = findZone(x, z);
                        if (zone) {{
                            zoneLabel.innerText = zone.label;
                            zoneLabel.style.color = zone.color;
                        }} else {{
                            zoneLabel.innerText = "Área Jugable";
                            zoneLabel.style.color = "#10b981";
                        }}
                    }} else {{
                        zoneLabel.innerText = "Fondo No Jugable";
                        zoneLabel.style.color = "#f43f5e";
                    }}

                    const soilLabel = document.getElementById('probe-soil');
                    const soil = soilAt(x, z);
                    if (soil) {{
                        soilLabel.innerText = `${{soil.name}} (${{soil.yield}})`;
                        soilLabel.style.color = soil.color;
                    }} else {{
                        soilLabel.innerText = "—";
                        soilLabel.style.color = "var(--text-muted)";
                    }}
                }} else {{
                    probePanel.style.display = 'none';
                }}
            }} else {{
                probePanel.style.display = 'none';
            }}
        }}

        function animate() {{
            requestAnimationFrame(animate);

            const now = performance.now();
            const dt = Math.min(0.1, (now - lastFrameTime) / 1000);
            lastFrameTime = now;

            if (povMode) {{
                updatePov(dt);
            }} else {{
                controls.update();
            }}

            checkTerrainIntersection();
            renderer.render(scene, camera);
        }}
    </script>
</body>
</html>
"""
    
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved HTML viewer to: {output_html_path}")
    print("\n=== Success! ===")
    print("To view the 3D visualization, start a local HTTP server in this directory:")
    print("  python3 -m http.server 8000")
    print("Then open in your browser:")
    print("  http://localhost:8000/dem_viewer_3d.html")

if __name__ == "__main__":
    main()
