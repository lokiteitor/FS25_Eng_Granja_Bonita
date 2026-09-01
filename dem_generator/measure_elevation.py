#!/usr/bin/env python3
"""Elevation report for the generated heightmap.

Checks the things that are easy to break and hard to see in a hillshade: that the river
bed really falls all the way downstream, that every farmyard is actually flat, and that
no field ended up on a slope nothing can drive on.
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_source as ms

FLAT_TOL_M = 0.25        # a farmyard should be level to within this
STEEP_DEG = 15.0         # farmland above this is awkward to work


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dem_path = os.path.join(script_dir, "dem_new_12k.png")
    if not os.path.exists(dem_path):
        print(f"Error: {dem_path} not found. Run generate_new_dem_12k.py first.")
        return

    raw = np.array(Image.open(dem_path), dtype=np.float32) / 100.0
    off, size = int(ms.OFFSET_M), int(ms.PLAYABLE_M)
    play = raw[off:off + size, off:off + size]

    print(f"=== Elevation report: {dem_path} ===")
    print(f"canvas   {raw.shape[1]}x{raw.shape[0]} px   "
          f"{raw.min():7.2f} .. {raw.max():7.2f} m")
    print(f"playable {size}x{size} m      {play.min():7.2f} .. {play.max():7.2f} m   "
          f"(relief {play.max() - play.min():.1f} m)")
    peak = np.unravel_index(int(np.argmax(play)), play.shape)
    print(f"highest point in play: {play[peak]:.2f} m at "
          f"X={peak[1]} Y={peak[0]}")

    # --- river profile
    # The lake reach is skipped: a lake is a basin, so the bed there dips well below the
    # channel either side of it and is meant to be non-monotonic.
    river = ms.load_river_path()
    lake = ms.lake_mask(size, 1.0)
    inside = [(x, y) for x, y in river if 0 <= x < size and 0 <= y < size]
    in_lake = np.array([bool(lake[int(y), int(x)]) for x, y in inside])
    bed = np.array([play[int(y), int(x)] for x, y in inside])
    # Diff within each contiguous run outside the lake: joining the reach above the lake
    # straight onto the one below it would compare two points a kilometre apart.
    runs = []
    start = None
    for i, wet in enumerate(in_lake):
        if not wet and start is None:
            start = i
        elif wet and start is not None:
            runs.append(bed[start:i])
            start = None
    if start is not None:
        runs.append(bed[start:])
    channel = bed[~in_lake]
    rises = np.concatenate([np.diff(r) for r in runs if len(r) > 1])
    print(f"\nriver: {len(inside)} samples inside the playable area "
          f"({int(in_lake.sum())} of them in the lake)")
    print(f"   channel {channel.max():.2f} -> {channel.min():.2f} m, "
          f"fall {channel.max() - channel.min():.2f} m")
    print(f"   largest step back uphill (lake reach excluded): "
          f"{max(rises.max(), 0.0):.3f} m"
          + ("" if rises.max() <= 0.05 else "   <-- the bed is not monotonic"))

    # --- lake
    z_lake = ms.lake_surface_z()
    z = play[lake]
    if z.size:
        depth = z_lake - z
        above = float((z > z_lake).sum()) / 10000.0
        shore = ndimage.binary_dilation(lake, ms.disk(25)) & ~lake
        chan_img = Image.new("L", (size, size), 0)
        ImageDraw.Draw(chan_img).line([(x, y) for x, y in river], fill=255,
                                      width=int(2 * (ms.RIVER_HALF_M + 14)),
                                      joint="curve")
        bank = shore & ~(np.array(chan_img) > 0)
        # Sub-water-line ground hugging the river is the inlet and the outlet, which are
        # meant to be below the lake: only a hollow well away from the water course is a
        # spill point.
        from scipy.spatial import cKDTree
        wet_bank = bank & (play < z_lake)
        low = float(wet_bank.sum()) / 10000.0
        far = 0.0
        if wet_bank.any():
            ys_b, xs_b = np.nonzero(wet_bank)
            d_riv = cKDTree(np.asarray(river)).query(
                np.column_stack([xs_b + 0.5, ys_b + 0.5]))[0]
            far = float((d_riv > 60.0).sum()) / 10000.0
        print(f"\nlake: {lake.sum() / 10000.0:.1f} ha at {z_lake:.2f} m, "
              f"bed {z.min():.2f} .. {z.max():.2f} m")
        print(f"   depth mean {depth.mean():.2f} m, max {depth.max():.2f} m")
        print(f"   ground inside standing out of the water: {above:.3f} ha"
              + ("" if above <= 0.02 else "   <-- shallows"))
        print(f"   bank below the water line: {low:.3f} ha, of which {far:.3f} ha is "
              f"more than 60 m from the river"
              + ("" if far <= 0.05 else "   <-- the lake would spill here"))

    # --- farmyards
    pads, _ = ms.settlement_pads(river)
    # The channel plus the shoulder of the levelling skirt: the DEM will not fill a
    # watercourse to flatten a yard, so the ground beside one is a ramp, not a pad.
    water = ms.river_corridor_mask(river, half_m=ms.RIVER_HALF_M + 26.0,
                                   n=size, s=1.0)
    print(f"\nfarmyards ({len(pads)}):")
    for pad in pads:
        img = Image.new("L", (size, size), 0)
        ImageDraw.Draw(img).polygon(pad["ring"], fill=255)
        # A pad that reaches the water keeps an unlevelled strip along the bank: the DEM
        # will not fill a channel to flatten a yard. That strip is not a defect, so it is
        # left out of the flatness figure and reported separately.
        area = np.array(img) > 0
        wet = area & water
        z = play[area & ~water]
        if not z.size:
            continue
        spread = float(z.max() - z.min())
        flag = "" if spread <= FLAT_TOL_M else "   <-- not level"
        note = f"   ({wet.sum() / 10000.0:.2f} ha of bank left alone)" if wet.any() else ""
        print(f"   {'VILLAGE' if pad['village'] else 'pad    '} {pad['ha']:5.1f} ha  "
              f"{np.median(z):6.2f} m   spread {spread:4.2f} m{flag}{note}")

    # --- farmland slopes
    osm_path = os.path.join(os.path.dirname(script_dir), "osm_generator", "map.osm")
    if not os.path.exists(osm_path):
        print("\n(no map.osm yet, skipping the farmland slope check)")
        return
    root = ET.parse(osm_path).getroot()
    nodes = {int(n.get('id')): ms.global_to_local(float(n.get('lat')),
                                                  float(n.get('lon')))
             for n in root.findall('node')}
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    n_fields = 0
    for way in root.findall('way'):
        tags = {t.get('k'): t.get('v') for t in way.findall('tag')}
        if tags.get('landuse') != 'farmland':
            continue
        draw.polygon([nodes[int(nd.get('ref'))] for nd in way.findall('nd')], fill=255)
        n_fields += 1
    mask = np.array(img) > 0
    gy, gx = np.gradient(play)          # 1 px = 1 m, so the gradient is already a slope
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))[mask]
    print(f"\nfarmland: {n_fields} fields, {mask.sum() / 10000.0:.0f} ha")
    print(f"   slope deg  median {np.median(slope):.1f}   p90 {np.percentile(slope, 90):.1f}"
          f"   p99 {np.percentile(slope, 99):.1f}   max {slope.max():.1f}")
    steep = (slope > STEEP_DEG).sum() / 10000.0
    print(f"   over {STEEP_DEG:.0f} deg: {steep:.2f} ha"
          + ("" if steep < 5.0 else "   <-- worth a look"))


if __name__ == "__main__":
    main()
