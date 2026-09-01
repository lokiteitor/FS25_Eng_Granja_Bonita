#!/usr/bin/env python3
"""Shared reading of the inspiration imagery, plus the geometry helpers built on it.

Both the DEM generator and the OSM generator need the same river alignment, the same
village and the same industrial pads: if each script traced them on its own, a tweak to
one threshold would silently move the flattened ground out from under the farmyards.
So everything derived from `inspiracion/` lives here and is imported by both.

The two source images are 1024x1024 and are mapped 1:1 onto the *playable* area, so one
image pixel is PLAYABLE_M / 1024 = 4 metres. The DEM's non-playable border is filled by
mirroring the image outwards (see `padded_height_trend`).

Local coordinates follow the convention the rest of the project already uses:
    playable metres, x east, y south from the north edge, origin at the NW corner.
"""
import math
import os
from collections import deque

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------------- geometry
CANVAS_M = 12288.0         # full heightmap canvas (1 px = 1 m)
PLAYABLE_M = 8192.0        # playable area, centred in the canvas
OFFSET_M = (CANVAS_M - PLAYABLE_M) / 2.0      # 2048 m

SRC_PX = 1024              # inspiration image size
MPP = PLAYABLE_M / SRC_PX  # metres per source pixel

# The reference images are mapped 1:1 onto the playable area, so every distance read out
# of them scales with the map: make the map twice as wide and the same photographed
# field is twice as wide on the ground. The tuning constants in the generators were set
# against a 4096 m map and are multiplied by this, so changing PLAYABLE_M above is all
# it takes to resize the whole thing.
#
# Physical widths - hedgerows, lane corridors, the smoothing kernel - are NOT scaled:
# a hedge is a hedge whatever the map measures.
TUNED_AT_M = 4096.0
MAP_SCALE = PLAYABLE_M / TUNED_AT_M

ROOT = os.path.dirname(os.path.abspath(__file__))
VISUAL_PATH = os.path.join(ROOT, "inspiracion", "mapa_visual.jpeg")
HEIGHT_PATH = os.path.join(ROOT, "inspiracion", "mapa_alturas.jpeg")

# Elevation design, in metres. English lowland: a broad valley with gentle uplands.
RIVER_Z_UPSTREAM = 52.0    # water surface where the river enters at the NW
RIVER_Z_DOWNSTREAM = 24.0  # ...and where it leaves at the SE
LAND_Z_MIN = 30.0          # lowest ground away from the river
LAND_Z_MAX = 110.0         # hilltops

RIVER_HALF_M = 16.0        # half-width of the water channel
RIVER_DEPTH_M = 3.5        # bed below the water surface

# A lake on the river, fed and drained by it. It is placed by position along the river
# rather than by map coordinates, so it stays on the water however the traced alignment
# moves. Across this reach the water surface is flat - that is what makes it a lake and
# not a wide bit of river - and the profile drops again at the outlet.
LAKE_FROM_T = 0.585        # fraction of river chainage where the lake begins
LAKE_TO_T = 0.715          # ...and where it ends
LAKE_HALF_MAX_M = 130.0 * MAP_SCALE   # widest half-width, at the middle of the reach
LAKE_DEPTH_M = 6.0         # deepest point below the water surface
LAKE_BANK_M = 4.0          # how far the shore climbs above the water at the rim
LAKE_SHELF_M = 0.45 * LAKE_HALF_MAX_M   # how far in the bed takes to reach full depth
LAKE_BANK_WIDTH_M = 0.35 * LAKE_HALF_MAX_M   # width of the graded shore outside it
LAKE_RIM = 1.35            # the bank is cut out to this multiple of the half-width
LAKE_FAR = 4.0             # sentinel for 'not in the lake reach' (finite, see lake_field)

# Working raster for every mask/cut pass, in metres per pixel. Two metres keeps an 8 m
# hedgerow four pixels wide, which is the narrowest thing any of the cutting passes has
# to resolve.
GRID_S = 2.0
GRID_N = int(PLAYABLE_M / GRID_S)

# A settlement smaller than this in the source is a single farmstead, not a place to put
# a production on. Scales with area, so it tracks the map size.
MIN_SETTLEMENT_HA = 1.0 * MAP_SCALE ** 2


# --------------------------------------------------------------------- helpers
def disk(radius_px):
    r = int(round(radius_px))
    if r < 1:
        return np.ones((1, 1), bool)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def simplify(points, tol):
    """Iterative Douglas-Peucker; the raster outlines are far too dense to keep."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return [tuple(p) for p in pts]
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a, b = pts[i0], pts[i1]
        seg = b - a
        seg_len = math.hypot(seg[0], seg[1])
        chunk = pts[i0 + 1:i1]
        if seg_len < 1e-9:
            d = np.hypot(chunk[:, 0] - a[0], chunk[:, 1] - a[1])
        else:
            d = np.abs(seg[0] * (chunk[:, 1] - a[1])
                       - seg[1] * (chunk[:, 0] - a[0])) / seg_len
        k = int(np.argmax(d))
        if d[k] > tol:
            k += i0 + 1
            keep[k] = True
            stack.append((i0, k))
            stack.append((k, i1))
    return [tuple(p) for p in pts[keep]]


def ring_area_ha(poly):
    return abs(sum(poly[k][0] * poly[k + 1][1] - poly[k + 1][0] * poly[k][1]
                   for k in range(len(poly) - 1))) / 2.0 / 10000.0


def polyline_length(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def resample(pts, step):
    p = np.asarray(pts, dtype=float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    tgt = np.arange(0.0, cum[-1], step)
    return np.column_stack([np.interp(tgt, cum, p[:, i]) for i in (0, 1)])


def chaikin(pts, iterations=2, closed=False):
    """Corner cutting. Turns a chain of Voronoi ridges into a lane that actually bends."""
    p = [tuple(q) for q in pts]
    for _ in range(iterations):
        if len(p) < 3:
            break
        out = [] if closed else [p[0]]
        rng = range(len(p)) if closed else range(len(p) - 1)
        for i in rng:
            a, b = p[i], p[(i + 1) % len(p)]
            out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        if not closed:
            out.append(p[-1])
        p = out
    return p


MIN_HOLE_PX = 30           # a smaller interior hole is raster noise, and gets filled


def open_holes(comp):
    """Cut a one-pixel slot from every interior hole out to the edge of the component.

    A way in the generated .osm is a simple polygon: it cannot carry a hole. Contouring
    a donut returns two rings, and emitting both gives a filled outer polygon with a
    filled inner one on top of it - so a field would cover the copse it was cut around,
    and overlap itself while doing it. Slotting each hole open turns the donut into one
    keyhole ring that traces correctly.

    Holes below MIN_HOLE_PX are filled rather than slotted: a single stray pixel in the
    middle of a lake is not a clearing, and slotting it draws a several-hundred-metre
    scar across the polygon to reach the shore.
    """
    from PIL import ImageDraw
    filled = ndimage.binary_fill_holes(comp)
    holes = filled & ~comp
    if not holes.any():
        return comp
    outside = ~filled
    if not outside.any():
        return comp

    lab, k = ndimage.label(holes, np.ones((3, 3), bool))
    sizes = ndimage.sum(holes, lab, range(1, k + 1))
    speck = np.nonzero(sizes < MIN_HOLE_PX)[0] + 1
    if len(speck):
        comp = comp | np.isin(lab, speck)
        if len(speck) == k:
            return comp

    # Distance to the nearest pixel outside the component, and which pixel that is.
    dist, idx = ndimage.distance_transform_edt(~outside, return_indices=True)
    img = Image.fromarray(comp.astype(np.uint8) * 255)
    draw = ImageDraw.Draw(img)
    for hid in range(1, k + 1):
        if sizes[hid - 1] < MIN_HOLE_PX:
            continue
        ys, xs = np.nonzero(lab == hid)
        j = int(np.argmin(dist[ys, xs]))
        y0, x0 = int(ys[j]), int(xs[j])
        y1, x1 = int(idx[0][y0, x0]), int(idx[1][y0, x0])
        draw.line([(x0, y0), (x1, y1)], fill=0, width=1)
    return np.array(img) > 0


def trace_components(mask, s, min_ha, simp_tol, offset=(0.0, 0.0), clip_to=None):
    """Outline every component of a boolean mask as simplified closed rings.

    Works on the bounding box of each component rather than the whole grid, which is
    what makes the per-parcel cutting passes affordable. Holes are slotted open first
    (see `open_holes`) so each component comes back as exactly one ring.
    """
    lab, k = ndimage.label(mask, np.ones((3, 3), bool))
    if k == 0:
        return []
    hi = PLAYABLE_M if clip_to is None else clip_to
    out = []
    objs = ndimage.find_objects(lab)
    for lid in range(1, k + 1):
        sl = objs[lid - 1]
        comp = lab[sl] == lid
        if comp.sum() * s * s / 10000.0 < min_ha:
            continue
        comp = open_holes(comp)
        h, w = comp.shape
        padded = np.zeros((h + 2, w + 2), dtype=np.float32)
        padded[1:-1, 1:-1] = comp
        ax_x = (np.arange(w + 2) - 1.5 + sl[1].start + 0.5) * s + offset[0]
        ax_y = (np.arange(h + 2) - 1.5 + sl[0].start + 0.5) * s + offset[1]
        gx, gy = np.meshgrid(ax_x, ax_y)
        fig, ax = plt.subplots()
        cs = ax.contour(gx, gy, padded, levels=[0.5])
        plt.close(fig)
        if not cs.allsegs[0]:
            continue
        rings = []
        for seg in cs.allsegs[0]:
            if len(seg) < 4:
                continue
            pts = [(min(max(float(x), 0.0), hi), min(max(float(y), 0.0), hi))
                   for x, y in seg]
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            pts = simplify(pts, simp_tol)
            if len(pts) < 4:
                continue
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            rings.append((pts, ring_area_ha(pts)))
        if not rings:
            continue
        # One component, one polygon: keep the enclosing ring and discard the rest.
        best = max(rings, key=lambda z: z[1])
        if best[1] >= min_ha:
            out.append(best)
    return out


def to_grid(mask_src, n=GRID_N):
    """Source-image mask -> working raster, nearest neighbour (indices must stay crisp)."""
    return np.array(Image.fromarray(mask_src.astype(np.uint8) * 255).resize(
        (n, n), Image.Resampling.NEAREST)) > 0


# --------------------------------------------------------------------- imagery
def load_visual():
    """Wood mask, building mask and hedgerow density, all at source-image resolution."""
    if not os.path.exists(VISUAL_PATH):
        raise SystemExit(f"Missing inspiration image: {VISUAL_PATH}")
    hsv = np.array(Image.open(VISUAL_PATH).convert("HSV"), dtype=np.float32)
    hue, sat, val = hsv[..., 0] / 255.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    val_s = ndimage.gaussian_filter(val, 1.0)

    # Woodland: dark, saturated and green.
    wood = (val < 0.32) & (sat > 0.22) & (hue > 0.18) & (hue < 0.47)
    wood = ndimage.binary_opening(wood, np.ones((3, 3), bool))
    wood = ndimage.binary_closing(wood, np.ones((7, 7), bool))

    # Buildings: desaturated and bright. The lanes are the same grey, so an opening
    # with a small disk drops the linear features and keeps the blobs; the closing
    # afterwards gathers a scatter of roofs into one settlement.
    roofs = (sat < 0.20) & (val > 0.33)
    blobs = ndimage.binary_opening(roofs, disk(2))
    blobs = ndimage.binary_closing(blobs, disk(6))
    blobs = ndimage.binary_opening(blobs, disk(2))
    setts = ndimage.binary_fill_holes(ndimage.binary_closing(blobs, disk(10)))

    # Hedgerow density: every dark line (hedge, tree belt) or grey line (lane). Blurred,
    # it is a good proxy for how small the fields are in each part of the map.
    barrier = ((val_s < 0.33) | (sat < 0.22)).astype(np.float32)
    density = ndimage.gaussian_filter(barrier, 30.0)
    return wood, setts, density


def load_height_trend():
    """The greyscale of the inspiration heightmap, at source-image resolution."""
    if not os.path.exists(HEIGHT_PATH):
        raise SystemExit(f"Missing inspiration image: {HEIGHT_PATH}")
    return np.array(Image.open(HEIGHT_PATH).convert("L"), dtype=np.float32)


def padded_height_trend(blur_px=9.0, pad_px=None):
    """Large-scale relief, mirrored outwards to cover the whole DEM canvas.

    The source is a stylised map in which every field is its own flat plateau and the
    lanes are drawn as bright lines; a heavy blur is what turns it back into a landform.
    """
    if pad_px is None:
        pad_px = int(round(OFFSET_M / MPP))
    a = load_height_trend()
    a = np.pad(a, pad_px, mode="reflect")
    return ndimage.gaussian_filter(a, blur_px)


RIVER_STEP = 8.0           # canonical spacing of the river centreline nodes
_RIVER_CACHE = {}


def load_river_path(step=None, extend_m=None):
    """The river centreline, as playable-area metres, running NW -> SE.

    Traced from the dark channel in the inspiration heightmap: threshold, keep the
    largest component, then walk its longest internal path with a double breadth-first
    search (the classic graph-diameter trick) and smooth the result.
    """
    # Cached and canonical. Sampling the same alignment at two different steps gives two
    # slightly different polylines, and anything derived from "the nearest river node" -
    # the lake half-width above all - then disagrees between the generators.
    if step is None:
        step = RIVER_STEP
    key = (step, extend_m)
    if key in _RIVER_CACHE:
        return _RIVER_CACHE[key]

    a = ndimage.median_filter(load_height_trend(), size=3)
    mask = ndimage.binary_closing(a < 55, structure=np.ones((5, 5), bool))
    lab, k = ndimage.label(mask, np.ones((3, 3), bool))
    if k == 0:
        raise SystemExit("No river found in the inspiration heightmap.")
    sizes = ndimage.sum(mask, lab, range(1, k + 1))
    comp = lab == (int(np.argmax(sizes)) + 1)
    h, w = comp.shape

    def bfs(src):
        seen = -np.ones(comp.shape, dtype=np.int32)
        parent = {}
        queue = deque([src])
        seen[src] = 0
        last = src
        while queue:
            y, x = queue.popleft()
            last = (y, x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and comp[ny, nx]
                            and seen[ny, nx] < 0):
                        seen[ny, nx] = seen[y, x] + 1
                        parent[(ny, nx)] = (y, x)
                        queue.append((ny, nx))
        return parent, last

    ys, xs = np.nonzero(comp)
    _, far = bfs((int(ys[0]), int(xs[0])))
    parent, other = bfs(far)
    path = [other]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path = path[::-1]
    if path[0][1] > path[-1][1]:
        path = path[::-1]

    pts = np.array([[x, y] for y, x in path], dtype=float) * MPP

    # The threshold picks up a pool at the south-eastern corner, and the traced path
    # doubles back through it. Cut at the point furthest downstream instead.
    cut = int(np.argmax(pts[:, 0] + pts[:, 1]))
    if cut > len(pts) * 0.5:
        pts = pts[:cut + 1]

    k_smooth = 41
    pad = np.vstack([np.repeat(pts[:1], k_smooth, 0), pts,
                     np.repeat(pts[-1:], k_smooth, 0)])
    sm = np.column_stack([
        np.convolve(pad[:, i], np.ones(k_smooth) / k_smooth, 'same')[k_smooth:-k_smooth]
        for i in (0, 1)])

    # Run both ends out past the border so the river leaves the map cleanly. It has to
    # clear the non-playable margin as well, or it simply stops in the backdrop.
    if extend_m is None:
        extend_m = OFFSET_M + 900.0
    d0 = sm[0] - sm[6]
    d0 /= max(np.hypot(*d0), 1e-9)
    d1 = sm[-1] - sm[-7]
    d1 /= max(np.hypot(*d1), 1e-9)
    sm = np.vstack([sm[0] + d0 * extend_m, sm, sm[-1] + d1 * extend_m])
    _RIVER_CACHE[key] = resample(sm, step)
    return _RIVER_CACHE[key]


def river_corridor_mask(river=None, half_m=None, n=GRID_N, s=GRID_S, with_lake=True):
    """Raster of the water channel plus a bank, on the working grid.

    The lake is part of the water: a farmyard levelled across it would drain it just as
    surely as one levelled across the channel.
    """
    from PIL import ImageDraw
    if river is None:
        river = load_river_path()
    if half_m is None:
        half_m = RIVER_HALF_M + 8.0
    img = Image.new("L", (n, n), 0)
    ImageDraw.Draw(img).line([(x / s, y / s) for x, y in river], fill=255,
                             width=max(1, int(round(2 * half_m / s))), joint="curve")
    mask = np.array(img) > 0
    if with_lake:
        mask |= ndimage.binary_dilation(lake_mask(n, s), disk(8.0 / s))
    return mask


def _hull_ring(mask, s, simp_tol=6.0):
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return None
    pts = np.column_stack([(xs + 0.5) * s, (ys + 0.5) * s])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    ring = [tuple(pts[i]) for i in hull.vertices]
    ring.append(ring[0])
    ring = simplify(ring, simp_tol)
    return ring if len(ring) >= 4 else None


def settlement_pads(river=None):
    """The village plus the outlying settlements, as compact convex pads.

    Every settlement becomes one flat farmyard: the village is the one nearest the
    centre of the map, the rest are the industrial pads. A convex hull is deliberate -
    the raw building clusters are ragged stars following the lanes, and a pad is only
    useful if you can actually drop a production on it.

    A pad may not straddle the river. Two of the settlements in the source sit on the
    bank, and their hull would otherwise reach across the water - which the DEM would
    then dutifully level, filling in the channel. Such a pad is cut back to whichever
    bank holds more of it, re-hulling until the hull itself clears the water.
    """
    from PIL import ImageDraw
    _, sett_src, _ = load_visual()
    grid = to_grid(sett_src)
    water = river_corridor_mask(river)

    lab, k = ndimage.label(grid, np.ones((3, 3), bool))
    pads = []
    n_clipped = 0
    for lid in range(1, k + 1):
        comp = lab == lid
        if comp.sum() * GRID_S * GRID_S / 10000.0 < MIN_SETTLEMENT_HA:
            continue
        ring = _hull_ring(comp, GRID_S)
        if ring is None:
            continue

        # The hull can swallow the river even when the buildings themselves do not, so
        # the check is on the hull, and the fix is applied to the hull's own footprint.
        keep = None
        for attempt in range(5):
            img = Image.new("L", (GRID_N, GRID_N), 0)
            ImageDraw.Draw(img).polygon([(x / GRID_S, y / GRID_S) for x, y in ring],
                                        fill=255, outline=255)
            filled = np.array(img) > 0
            if not (filled & water).any():
                break
            if attempt == 0:
                n_clipped += 1
            sub, ns = ndimage.label(filled & ~water, np.ones((3, 3), bool))
            if ns == 0:
                ring = None
                break
            sizes = ndimage.sum(filled & ~water, sub, range(1, ns + 1))
            keep = sub == (int(np.argmax(sizes)) + 1)
            new_ring = _hull_ring(keep, GRID_S)
            if new_ring is None:
                ring = None
                break
            ring = new_ring
        else:
            # Re-hulling keeps reaching back over the water where the river bends
            # through the settlement. Fall back to the clipped outline itself: a
            # slightly concave pad still takes a production, a flooded one does not.
            # The Douglas-Peucker pass inside trace_components can cut a corner back
            # across the bank, so the footprint is eroded by that much slack first.
            if keep is not None:
                keep = ndimage.binary_erosion(keep, disk(3.0 / GRID_S))
            traced = trace_components(keep, GRID_S, 1.0, 3.0) if keep is not None \
                and keep.any() else []
            ring = max(traced, key=lambda z: z[1])[0] if traced else None
        if ring is None:
            continue
        area = ring_area_ha(ring)
        if area < MIN_SETTLEMENT_HA:
            continue
        cx = sum(p[0] for p in ring[:-1]) / (len(ring) - 1)
        cy = sum(p[1] for p in ring[:-1]) / (len(ring) - 1)
        pads.append({"ring": ring, "ha": area, "cx": cx, "cy": cy, "village": False})

    if not pads:
        raise SystemExit("No settlements found in the inspiration image.")
    if n_clipped:
        print(f"   clipped {n_clipped} settlement hull(s) back off the river")
    pads.sort(key=lambda p: -p["ha"])
    village = min(pads, key=lambda p: math.hypot(p["cx"] - PLAYABLE_M / 2.0,
                                                 p["cy"] - PLAYABLE_M / 2.0))
    village["village"] = True
    return pads, village


def grow_ring(ring, metres):
    """Push a convex ring outwards from its centroid by roughly `metres`."""
    cx = sum(p[0] for p in ring[:-1]) / (len(ring) - 1)
    cy = sum(p[1] for p in ring[:-1]) / (len(ring) - 1)
    out = []
    for x, y in ring[:-1]:
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy)
        if d < 1e-6:
            out.append((x, y))
            continue
        f = (d + metres) / d
        out.append((min(max(cx + dx * f, 0.0), PLAYABLE_M),
                    min(max(cy + dy * f, 0.0), PLAYABLE_M)))
    out.append(out[0])
    return out


def lake_surface_z():
    """The level the lake sits at: what the river surface would be at the middle of the
    reach if it fell uniformly."""
    t = (LAKE_FROM_T + LAKE_TO_T) / 2.0
    return RIVER_Z_UPSTREAM + (RIVER_Z_DOWNSTREAM - RIVER_Z_UPSTREAM) * t


def river_surface_z(chainage, total):
    """Water surface elevation: falling from the NW entry to the SE exit, but held flat
    across the lake reach. Still monotonically non-increasing downstream."""
    z = lake_surface_z()
    return np.interp(chainage,
                     [0.0, LAKE_FROM_T * total, LAKE_TO_T * total, total],
                     [RIVER_Z_UPSTREAM, z, z, RIVER_Z_DOWNSTREAM])


def _lake_half_width(u, side):
    """Half-width across the lake, u in [0, 1] along the reach, side -1/+1 across it.

    The two banks get different wobble, so the lake is an irregular basin rather than a
    symmetric lens pasted over the river.
    """
    uu = np.clip(u, 0.0, 1.0)
    base = np.sin(np.pi * uu) ** 0.65
    left = 1.0 + 0.24 * np.sin(5.3 * np.pi * uu + 1.1) \
        + 0.13 * np.sin(9.7 * np.pi * uu + 0.3)
    right = 1.0 + 0.20 * np.sin(4.1 * np.pi * uu + 2.7) \
        + 0.15 * np.sin(8.3 * np.pi * uu + 1.9)
    return LAKE_HALF_MAX_M * base * np.where(side >= 0, left, right)


def river_tangents(river):
    tang = np.gradient(np.asarray(river, dtype=float), axis=0)
    return tang / np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-9)[:, None]


def lake_field(query_xy, nearest_idx, nearest_dist, river, chain, total, tangents=None):
    """For each query point, its position along the lake reach and its distance from the
    centreline as a fraction of the local half-width.

    `r <= 1` is open water; `1 < r <= LAKE_RIM` is the bank the DEM cuts down to the
    shore. Points whose nearest river node is outside the reach get `r = inf`.
    """
    if tangents is None:
        tangents = river_tangents(river)
    c0 = LAKE_FROM_T * total
    c1 = LAKE_TO_T * total
    u = (chain[nearest_idx] - c0) / max(c1 - c0, 1e-9)
    rel = np.asarray(query_xy, dtype=float) - np.asarray(river)[nearest_idx]
    t_at = tangents[nearest_idx]
    side = np.sign(t_at[:, 0] * rel[:, 1] - t_at[:, 1] * rel[:, 0])
    half = _lake_half_width(u, side)
    in_reach = (u >= 0.0) & (u <= 1.0) & (half > 1e-6)
    # A finite sentinel, not inf: the DEM evaluates this on a coarse grid and scales the
    # result up bilinearly, and interpolating against infinity poisons the shore.
    r = np.where(in_reach, nearest_dist / np.maximum(half, 1e-6), LAKE_FAR)
    return u, np.minimum(r, LAKE_FAR)


_LAKE_CACHE = {}


def lake_r_field(margin_m=None):
    """The lake's `r` field at one metre, over its own bounding box.

    Both generators read the lake from this single array. Evaluating the field twice on
    different grids does not work: inside the lake the river meanders, and a pixel's
    nearest river node - and with it the local half-width - flips between limbs from one
    grid to the next, which puts the excavated basin and the drawn shoreline metres
    apart in places.

    Returns `(r, x0, y0)`, where `r[i, j]` is the value at playable metres
    `(x0 + j + 0.5, y0 + i + 0.5)`.
    """
    from scipy.spatial import cKDTree
    if 'r' in _LAKE_CACHE:
        return _LAKE_CACHE['r']
    river = np.asarray(load_river_path(), dtype=float)
    seg = np.hypot(*np.diff(river, axis=0).T)
    chain = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(chain[-1])

    if margin_m is None:
        margin_m = LAKE_HALF_MAX_M * LAKE_RIM + 60.0
    reach = river[(chain >= LAKE_FROM_T * total) & (chain <= LAKE_TO_T * total)]
    x0 = int(math.floor(reach[:, 0].min() - margin_m))
    y0 = int(math.floor(reach[:, 1].min() - margin_m))
    x1 = int(math.ceil(reach[:, 0].max() + margin_m))
    y1 = int(math.ceil(reach[:, 1].max() + margin_m))

    gy, gx = np.mgrid[y0:y1, x0:x1]
    query = np.column_stack([gx.ravel() + 0.5, gy.ravel() + 0.5])
    dist, idx = cKDTree(river).query(query, workers=-1)
    _, r = lake_field(query, idx, dist, river, chain, total)
    _LAKE_CACHE['r'] = (r.reshape(y1 - y0, x1 - x0).astype(np.float32), x0, y0)
    return _LAKE_CACHE['r']


def lake_mask(n, s, origin=0.0):
    """Open-water mask of the lake on an n x n grid at s metres per pixel, sampled from
    the shared one-metre field.

    `origin` shifts the grid into the river's frame (the DEM works in canvas metres,
    the OSM in playable metres).
    """
    r, x0, y0 = lake_r_field()
    out = np.zeros((n, n), dtype=bool)
    gy, gx = np.mgrid[0:n, 0:n]
    # floor, not truncation: astype(int) rounds towards zero, so a coordinate just left
    # of the field's origin would alias onto column 0 instead of falling outside it.
    px = np.floor(gx * s + s / 2.0 + origin - x0).astype(np.int64)
    py = np.floor(gy * s + s / 2.0 + origin - y0).astype(np.int64)
    ok = (px >= 0) & (px < r.shape[1]) & (py >= 0) & (py < r.shape[0])
    out[ok] = r[py[ok], px[ok]] <= 1.0
    return out



# --------------------------------------------------------------------- projection
# Map centre. English lowland farmland (north Oxfordshire); the exact spot only matters
# for how the .osm reads in an editor, but it keeps latitudes and field shapes sane.
LAT_CENTER = 52.0620
LON_CENTER = -1.3400

_M_PER_DEG_LAT = 111111.0
_M_PER_DEG_LON = 111111.0 * math.cos(math.radians(LAT_CENTER))


def local_to_global(x, y):
    """Playable metres (x east, y south from the north edge) -> lat/lon."""
    return (LAT_CENTER + (PLAYABLE_M / 2.0 - y) / _M_PER_DEG_LAT,
            LON_CENTER + (x - PLAYABLE_M / 2.0) / _M_PER_DEG_LON)


def global_to_local(lat, lon):
    """Inverse of `local_to_global`."""
    return ((lon - LON_CENTER) * _M_PER_DEG_LON + PLAYABLE_M / 2.0,
            PLAYABLE_M / 2.0 - (lat - LAT_CENTER) * _M_PER_DEG_LAT)
