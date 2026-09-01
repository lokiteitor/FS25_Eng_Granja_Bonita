#!/usr/bin/env python3
"""FS25 heightmap generator - English lowland river valley.

Builds a 6144x6144 m canvas (1 px = 1 m) with the 4096x4096 m playable area centred in
it, from the two images in `inspiracion/`:

  * `mapa_alturas.jpeg` is a stylised map in which every field is its own flat plateau
    and the lanes are drawn as bright lines. Blurred hard it stops being a field map and
    becomes the landform: a valley running NW -> SE with gentle uplands either side.
    That is the large-scale trend the terrain is built on.
  * the dark channel in the same image is the river, traced in `map_source.py` and carved
    here as a proper valley with a water surface that falls steadily downstream.

On top of that go a little rolling noise, and the flat pads the OSM generator puts the
village and the industrial farmyards on. Those pads come from `map_source.py` too, so the
flattened ground and the farmyard polygons cannot drift apart.

Heights are stored as 16-bit centimetres (raw / 100 = metres), matching the rest of the
project and Giants Editor's import convention.
"""
import os
import sys
import time
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy import ndimage
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_source as ms

# The river field is evaluated on a coarse grid and scaled up: the distance and chainage
# fields are already far smoother than the metre they are resampled to.
FIELD_S = 4.0

# Distances read off the reference image scale with the map (see map_source.MAP_SCALE);
# physical widths do not.
K = ms.MAP_SCALE

# How hard the stylised source is blurred before it is used as a landform. The plateaus
# in it are 50-150 m across at the tuning size, so anything much less leaves the field
# pattern embossed in the terrain as lumps.
TREND_BLUR_M = 112.0 * K

# Distance from the channel -> height above the local water surface. The flat run inside
# VALLEY_FULL_M is the water meadow; past that the ground climbs out of the valley.
VALLEY_KNOTS_M = [0.0, ms.RIVER_HALF_M, 30.0 * K, 110.0 * K, 300.0 * K, 480.0 * K]
VALLEY_RISE_M = [-ms.RIVER_DEPTH_M, -ms.RIVER_DEPTH_M, 1.5, 3.0, 10.0, 22.0]
VALLEY_FULL_M = 110.0 * K  # inside this the valley profile is used as-is
VALLEY_FADE_M = 480.0 * K  # ...and beyond this only the regional trend remains

LAKE_FREEBOARD_M = 0.6     # how far the shore stands proud of the water line
PAD_SKIRT_M = 55.0 * K     # cosine ramp around every flattened farmyard
PAD_MARGIN_M = 20.0 * K    # pads are flattened a little wider than the OSM polygon
SMOOTH_SIGMA_M = 3.0       # physical: the terrain is metre-resolution either way


def val_noise(shape, grid_size, weight, seed):
    """Smooth value noise: a small random grid blown up with bicubic interpolation."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(-1.0, 1.0, size=(grid_size, grid_size)).astype(np.float32)
    img = Image.fromarray(small).resize((shape[1], shape[0]), Image.Resampling.BICUBIC)
    return np.array(img, dtype=np.float32) * weight


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def flatten_pad(terrain, ring, skirt, label, protect=None):
    """Level the ground under a farmyard, with a cosine skirt into the surroundings.

    The pad follows the polygon itself, not its bounding box: these are convex hulls of
    a settlement and squaring them off would flatten a lot of countryside that is meant
    to stay rolling.
    """
    from PIL import ImageDraw
    n = terrain.shape[0]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    bx0 = max(0, int(min(xs) - skirt - 6))
    bx1 = min(n - 1, int(max(xs) + skirt + 6))
    by0 = max(0, int(min(ys) - skirt - 6))
    by1 = min(n - 1, int(max(ys) + skirt + 6))
    if bx1 - bx0 < 4 or by1 - by0 < 4:
        return None

    w_px, h_px = bx1 - bx0 + 1, by1 - by0 + 1
    img = Image.new("L", (w_px, h_px), 0)
    ImageDraw.Draw(img).polygon([(x - bx0, y - by0) for x, y in ring],
                                fill=255, outline=255)
    inside = np.array(img) > 0
    if protect is not None:
        # The pads are already clipped off the river in map_source, but they are grown
        # by a margin here, which can reach back over the bank. The channel wins.
        inside &= ~protect[by0:by1 + 1, bx0:bx1 + 1]
    if not inside.any():
        return None

    # 1 px = 1 m, so the transform is already in metres.
    dist = ndimage.distance_transform_edt(~inside).astype(np.float32)

    patch = terrain[by0:by1 + 1, bx0:bx1 + 1]
    target = float(np.median(patch[inside]))
    ramp = (dist > 0.0) & (dist <= skirt)
    if protect is not None:
        ramp &= ~protect[by0:by1 + 1, bx0:bx1 + 1]
    ref = patch.copy()
    patch[inside] = target
    w = 0.5 * (1.0 + np.cos(np.pi * dist[ramp] / skirt))
    patch[ramp] = w * target + (1.0 - w) * ref[ramp]
    # Re-smooth only the ramp, so the pad edge does not read as a crease.
    local = ndimage.gaussian_filter(patch, sigma=6.0)
    patch[ramp] = local[ramp]
    terrain[by0:by1 + 1, bx0:bx1 + 1] = patch
    print(f"   {label}: levelled to {target:.1f} m "
          f"({inside.sum()/10000.0:.1f} ha flat)")
    return target


def main():
    t_start = time.time()
    n = int(ms.CANVAS_M)
    print(f"=== FS25 English DEM generator ({n}x{n} m canvas, "
          f"{ms.PLAYABLE_M:.0f} m playable) ===")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dem = os.path.join(script_dir, "dem_new_12k.png")
    out_vis = os.path.join(script_dir, "dem_new_visual_12k.png")
    out_detail = os.path.join(script_dir, "dem_new_visual_detail_12k.png")

    print("1. Regional trend from 'inspiracion/mapa_alturas.jpeg'...")
    trend_src = ms.padded_height_trend(blur_px=TREND_BLUR_M / ms.MPP)
    trend = np.array(Image.fromarray(trend_src).resize((n, n),
                     Image.Resampling.BICUBIC), dtype=np.float32)
    lo, hi = np.percentile(trend, 0.5), np.percentile(trend, 99.5)
    trend = np.clip((trend - lo) / (hi - lo), 0.0, 1.0)
    trend = ms.LAND_Z_MIN + trend * (ms.LAND_Z_MAX - ms.LAND_Z_MIN)
    print(f"   trend {trend.min():.1f} .. {trend.max():.1f} m")

    print("2. Rolling relief noise...")
    noise = (val_noise((n, n), 12, 1.00, 20260901)
             + val_noise((n, n), 28, 0.45, 20260902)
             + val_noise((n, n), 64, 0.18, 20260903))
    noise *= 3.2 / max(float(np.abs(noise).max()), 1e-6)
    terrain = trend + noise

    print("3. Carving the river valley...")
    river = ms.load_river_path()
    river_canvas = river + ms.OFFSET_M
    seg = np.hypot(*np.diff(river_canvas, axis=0).T)
    chain = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(chain[-1])
    print(f"   river: {len(river_canvas)} nodes, {total/1000.0:.2f} km, "
          f"{ms.RIVER_Z_UPSTREAM:.0f} m -> {ms.RIVER_Z_DOWNSTREAM:.0f} m "
          f"({(ms.RIVER_Z_UPSTREAM-ms.RIVER_Z_DOWNSTREAM)/(total/1000.0):.1f} m/km)")

    fn = int(ms.CANVAS_M / FIELD_S)
    gy, gx = np.mgrid[0:fn, 0:fn]
    query = np.column_stack([((gx.ravel() + 0.5) * FIELD_S),
                             ((gy.ravel() + 0.5) * FIELD_S)])
    # The lowest of the nearest few river points, not simply the nearest: at a meander
    # neck the closest point can belong to the other limb, several hundred metres of
    # chainage away, and the bed would step back uphill there. Taking the minimum makes
    # the neck a touch deeper instead, which is where a cutoff would form anyway.
    dist_k, idx_k = cKDTree(river_canvas).query(query, k=12, workers=-1)
    dist_f = dist_k[:, 0].reshape(fn, fn).astype(np.float32)
    surf_f = ms.river_surface_z(chain[idx_k], total).min(axis=1)
    surf_f = surf_f.reshape(fn, fn).astype(np.float32)

    def upscale(a):
        return np.array(Image.fromarray(a).resize((n, n), Image.Resampling.BILINEAR),
                        dtype=np.float32)

    dist_m = upscale(dist_f)
    surface = upscale(surf_f)

    valley = surface + np.interp(dist_m, VALLEY_KNOTS_M, VALLEY_RISE_M).astype(np.float32)
    blend = 1.0 - smoothstep((dist_m - VALLEY_FULL_M) / (VALLEY_FADE_M - VALLEY_FULL_M))
    terrain = blend * valley + (1.0 - blend) * terrain

    print("3b. Excavating the lake...")
    # The lake is worked out at one metre over its own bounding box, from the same shared
    # field the OSM traces its shoreline from - see map_source.lake_r_field. `r` is the
    # distance from the centreline as a fraction of the local half-width: r <= 1 is open
    # water, out to LAKE_RIM is the bank that is cut down to meet it.
    r_lake, lx0, ly0 = ms.lake_r_field()
    lh, lw = r_lake.shape
    cy0, cx0 = int(ly0 + ms.OFFSET_M), int(lx0 + ms.OFFSET_M)
    ty0, tx0 = max(0, cy0), max(0, cx0)
    ty1, tx1 = min(n, cy0 + lh), min(n, cx0 + lw)
    win = (slice(ty0, ty1), slice(tx0, tx1))
    r_m = r_lake[ty0 - cy0:ty1 - cy0, tx0 - cx0:tx1 - cx0]
    z_lake = ms.lake_surface_z()

    # The bed and the shore are shaped by distance from the shoreline, not by `r`.
    # `r` is a fine test for what is water, but it is not continuous: inside the lake the
    # river meanders, and along the ridge between two limbs the nearest river node - and
    # with it the local half-width - flips, so `r` steps. Shaping the ground with it puts
    # a cliff down the middle of the lake. A distance transform has no such seam.
    water_m = r_m <= 1.0
    d_in = ndimage.distance_transform_edt(water_m).astype(np.float32)
    d_out = ndimage.distance_transform_edt(~water_m).astype(np.float32)

    # Bed: a shelf off the shore, flattening out to full depth further in.
    t_in = np.clip(d_in / ms.LAKE_SHELF_M, 0.0, 1.0)
    bowl = (z_lake - ms.LAKE_DEPTH_M * np.sqrt(t_in)).astype(np.float32)

    # Shore: climbs to LAKE_BANK_M over LAKE_BANK_WIDTH_M, and the weight with which the
    # ground is moved onto it falls to zero over the same distance. One blend does both
    # jobs - it cuts a hillside that runs straight into the water down to a shore you can
    # walk, and it lifts any hollow behind the bank that the lake would otherwise drain
    # into - and because it fades out, the worked ground meets the untouched hillside
    # flush instead of over a wall.
    t_out = np.clip(d_out / ms.LAKE_BANK_WIDTH_M, 0.0, 1.0)
    bank = (z_lake + LAKE_FREEBOARD_M + ms.LAKE_BANK_M * t_out).astype(np.float32)
    w_band = (1.0 - t_out).astype(np.float32)
    band = (~water_m) & (d_out <= ms.LAKE_BANK_WIDTH_M)
    # Never raise ground inside the river channel: that would dam the inlet and outlet.
    in_channel = dist_m[win] <= ms.RIVER_HALF_M + 12.0

    def lake_blend(pat):
        pat[water_m] = np.minimum(pat[water_m], bowl[water_m])
        move = w_band * (bank - pat)
        move = np.where(in_channel, np.minimum(move, 0.0), move)
        pat[band] += move[band]

    patch = terrain[win]
    lake_blend(patch)
    terrain[win] = patch
    carve = water_m | band

    lake_water = np.zeros((n, n), dtype=bool)
    lake_water[win] = r_m <= 1.0
    lake_rim = np.zeros((n, n), dtype=bool)
    lake_rim[win] = carve
    water = lake_water
    print(f"   lake: {water.sum() / 10000.0:.1f} ha of open water at {z_lake:.2f} m, "
          f"{ms.LAKE_DEPTH_M:.1f} m deep, "
          f"river {ms.RIVER_Z_UPSTREAM:.0f} -> {z_lake:.1f} -> "
          f"{ms.RIVER_Z_DOWNSTREAM:.0f} m")

    print("4. Levelling the village and the industrial pads...")
    channel = (dist_m <= ms.RIVER_HALF_M + 6.0) | lake_rim
    pads, village = ms.settlement_pads(river)
    for i, pad in enumerate(pads):
        ring = [(x + ms.OFFSET_M, y + ms.OFFSET_M)
                for x, y in ms.grow_ring(pad["ring"], PAD_MARGIN_M)]
        label = "Village" if pad["village"] else f"Industry pad {i}"
        flatten_pad(terrain, ring, PAD_SKIRT_M, f"{label} ({pad['ha']:.1f} ha)",
                    protect=channel)

    print("5. Final smoothing and clamping...")
    terrain = ndimage.gaussian_filter(terrain, sigma=SMOOTH_SIGMA_M)

    # Re-assert the lake. The global blur smears the shoreline step, which leaves
    # shallows standing out of the water inside the polygon and hollows below the water
    # line just outside it. The bowl and the bank are smooth analytic surfaces, so
    # stamping them back on costs the terrain nothing and makes the basin exact.
    patch = terrain[win]
    lake_blend(patch)
    terrain[win] = patch
    raw = np.clip(terrain * 100.0, 2000.0, 62000.0)

    print(f"6. Writing '{os.path.basename(out_dem)}'...")
    Image.fromarray(raw.astype(np.int32), mode="I").save(out_dem)
    play = raw[int(ms.OFFSET_M):int(ms.OFFSET_M + ms.PLAYABLE_M),
               int(ms.OFFSET_M):int(ms.OFFSET_M + ms.PLAYABLE_M)]
    print(f"   canvas   {raw.min()/100:.2f} .. {raw.max()/100:.2f} m")
    print(f"   playable {play.min()/100:.2f} .. {play.max()/100:.2f} m "
          f"(relief {(play.max()-play.min())/100:.1f} m)")

    print("7. Visualisations...")
    vis_scale = max(1, n // 1024)
    vis = raw[::vis_scale, ::vis_scale]
    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(vis, cmap=plt.get_cmap('terrain'), vert_exag=0.55,
                      blend_mode='overlay')
    levels = [3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0]

    def style(ax, title):
        ax.set_xlabel("X (East-West) [metres]", fontsize=11, fontweight='bold')
        ax.set_ylabel("Y (North-South) [metres]", fontsize=11, fontweight='bold')
        ax.grid(True, which='both', color='white', linestyle='--', linewidth=0.5,
                alpha=0.35)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=14, color='white')

    # --- full canvas
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.imshow(shaded, extent=[0, n, n, 0])
    ax.set_xticks(np.arange(0, n + 1, 1024))
    ax.set_yticks(np.arange(0, n + 1, 1024))
    style(ax, f"Full DEM canvas ({n}x{n} px, 1 px = 1 m)")
    rect = plt.Rectangle((ms.OFFSET_M, ms.OFFSET_M), ms.PLAYABLE_M, ms.PLAYABLE_M,
                         fill=False, edgecolor='white', linewidth=2, linestyle='--',
                         label=f'Playable border ({ms.PLAYABLE_M/1000:.1f} km)')
    ax.add_patch(rect)
    ax.plot(river_canvas[:, 0], river_canvas[:, 1], color='#38bdf8', linewidth=1.6,
            label='River')
    axis = np.arange(vis.shape[0]) * vis_scale
    gxv, gyv = np.meshgrid(axis, axis)
    cnt = ax.contour(gxv, gyv, vis, levels=levels, colors=['#84cc16'],
                     linewidths=0.8, alpha=0.75)
    ax.clabel(cnt, inline=True, fmt=lambda v: f"{int(v/100)}m", fontsize=6,
              colors='#84cc16')
    ax.legend(loc='upper right', facecolor='black', labelcolor='white', fontsize=9)
    plt.savefig(out_vis, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"   {out_vis}")

    # --- playable area only
    p0 = int(ms.OFFSET_M / vis_scale)
    p1 = int((ms.OFFSET_M + ms.PLAYABLE_M) / vis_scale)
    sub_shape = p1 - p0
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.imshow(shaded[p0:p1, p0:p1], extent=[0, ms.PLAYABLE_M, ms.PLAYABLE_M, 0])
    ax.set_xticks(np.arange(0, ms.PLAYABLE_M + 1, 1024))
    ax.set_yticks(np.arange(0, ms.PLAYABLE_M + 1, 1024))
    style(ax, f"Playable area ({ms.PLAYABLE_M/1000:.1f} x {ms.PLAYABLE_M/1000:.1f} km)")
    ax.plot(river[:, 0], river[:, 1], color='#38bdf8', linewidth=2.2)
    lake_play = water[int(ms.OFFSET_M)::vis_scale, int(ms.OFFSET_M)::vis_scale]
    lake_play = lake_play[:sub_shape, :sub_shape]
    ax.contour(np.arange(sub_shape) * vis_scale, np.arange(sub_shape) * vis_scale,
               lake_play.astype(float), levels=[0.5], colors=['#0ea5e9'], linewidths=2.0)
    for pad in pads:
        ring = np.array(pad["ring"])
        colour = '#f472b6' if not pad["village"] else '#a78bfa'
        ax.plot(ring[:, 0], ring[:, 1], color=colour, linewidth=1.8)
        ax.text(pad["cx"], pad["cy"],
                "VILLAGE" if pad["village"] else f"{pad['ha']:.0f} ha",
                color=colour, fontsize=7, fontweight='bold', ha='center')
    sub = vis[p0:p1, p0:p1]
    axis = np.arange(sub.shape[0]) * vis_scale
    gxv, gyv = np.meshgrid(axis, axis)
    cnt = ax.contour(gxv, gyv, sub, levels=levels, colors=['#84cc16'],
                     linewidths=1.0, alpha=0.8)
    ax.clabel(cnt, inline=True, fmt=lambda v: f"{int(v/100)}m", fontsize=7,
              colors='#84cc16')
    ax.set_xlim(0, ms.PLAYABLE_M)
    ax.set_ylim(ms.PLAYABLE_M, 0)
    plt.savefig(out_detail, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"   {out_detail}")

    print(f"\n=== Done in {time.time() - t_start:.1f} s ===")


if __name__ == "__main__":
    main()
