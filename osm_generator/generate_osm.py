#!/usr/bin/env python3
"""FS25 OSM generator - English bocage countryside.

Writes `map.osm` for the playable 8192 x 8192 m area, laid out from the two images in
`inspiracion/` (read through `map_source.py`, shared with the DEM generator):

  * woodland, buildings and hedgerow density come from `mapa_visual.jpeg`;
  * the river alignment comes from the dark channel in `mapa_alturas.jpeg`.

The field pattern is not traced pixel-by-pixel - the hedges in the source photo are not a
closed partition, so tracing them merges whole districts into single 100 ha blobs.
Instead the *distribution* is followed: the local hedgerow density sets a target field
size, Poisson-disk seeds are drawn at that spacing, and their Voronoi cells become the
parcels. Dense hedges near the village give small fields, the open uplands give large
ones, which is what the source actually shows.

The lanes are chosen from the Voronoi edges themselves - a shortest-path tree over the
cell boundaries linking the village, the settlements and the map edges - so they run
along field boundaries and wander the way English lanes do, rather than on a grid.

Settlements: the one nearest the centre of the map is the village, emitted as a single
`landuse=farmyard` with its streets on top. Every other settlement becomes one flat
`landuse=farmyard` pad for a production to be placed on, per the map brief.

The lanes know nothing about the village, so they are trimmed at its boundary and the
streets inside are laid out from the entrances that leaves: a high street between the
two most opposite ones, a side street per remaining entrance, and a back lane closing a
block either side. Without that the village is simply the two networks overlaid.

Tag vocabulary is unchanged from the previous map, so the rest of the pipeline
(visualizer, Giants Editor import) keeps working:
    landuse=farmland                              fields
    landuse=farmyard                              village, industry pads, yards
    natural=wood + landuse=farmyard + leaf_type   woodland
    natural=water                                 the river
    highway=primary / secondary / tertiary        road hierarchy
"""
import heapq
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import Voronoi, cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_source as ms

SEED = 2026

# Distances read off the reference image scale with the map (see map_source.MAP_SCALE);
# physical widths - hedgerows, lane corridors - do not.
K = ms.MAP_SCALE

# --- field sizing -----------------------------------------------------------------
# Target spacing between field seeds, from the local hedgerow density. Poisson-disk
# sampling settles at roughly 0.69 points per r^2, so the spacing is scaled to land on
# the parcel count the brief asks for rather than used raw.
SPACING_OPEN_M = 470.0 * K   # sparse hedges -> big upland fields
SPACING_DENSE_M = 140.0 * K  # dense hedges -> small fields near the village
SPACING_SCALE = 0.66
MAX_FIELDS = 200             # hard cap from the map brief

HEDGE_M = 4.0                # physical: each field stands back this far -> 8 m hedges
WOOD_MARGIN_M = 10.0         # physical: ...and a little further from a wood
LANE_HALF_M = 8.0            # physical: corridor either side of a lane centreline

# Outline tolerances are physical too: a field boundary wants to be right to a few
# metres however big the map is. Every mask a parcel is cut against is grown by
# SIMPLIFY_SLACK_M first, because the Douglas-Peucker pass that follows can pull an
# edge back across the boundary it was just cut to - which is how fields end up
# overlapping the woods they were supposed to stop at.
FIELD_SIMPLIFY_M = 6.0
WOOD_SIMPLIFY_M = 8.0
SIMPLIFY_SLACK_M = 12.0
MIN_FIELD_HA = 1.2 * K * K
MIN_FIELD_WIDTH_M = 45.0 * K  # 2*area/perimeter; drops slivers pinched off by a lane
MIN_WOOD_HA = 0.8 * K * K

# --- lanes ------------------------------------------------------------------------
# The crossing penalties are added to edge lengths, which scale with the map, so they
# scale too or the network would start fording the river wherever it fancied.
RIVER_CROSS_PENALTY = 3000.0 * K   # keeps crossings down to a handful of bridges
WOOD_CROSS_PENALTY = 1200.0 * K
LANE_COVERAGE_M = 500.0            # physical: how far you may drive off-road to a field
                                   # (a gameplay distance, so it does not scale)
LANE_SMOOTH_ITERS = 2
STUB_MIN_M = 170.0 * K             # shorter dead-end lanes are offcuts, not tracks
PAD_LINK_MAX_M = 400.0 * K         # furthest a farmyard access spur will reach
BRIDGE_HALF_M = ms.RIVER_HALF_M + 20.0   # within this of the centreline is the bridge

# --- village streets --------------------------------------------------------------
# The village is laid out, not accumulated: one high street between the two most
# opposite entrances, a side street from each of the others joining it at its own
# T-junction, and one back lane closing a block. These are proportions of the pad, so
# they scale with it.
GATE_MERGE_M = 80.0 * K       # lanes arriving this close share one entrance
STREET_MIN_GAP_M = 45.0 * K   # T-junctions on the high street stand at least this apart
                              # (small on purpose: an 800 m street already carrying the
                              # two ends of the green has no room for a wide gap, and
                              # asking for one only pushes a side street to the far end
                              # of the village, away from the entrance it serves)
BACK_LANE_OFF_M = 110.0 * K   # how far the back lane stands off the high street
STREET_INSET_M = 70.0 * K     # the perimeter lane runs this far inside the boundary
CROSS_SPACING_M = 70.0 * K    # spacing of the cross streets along the green
CROSS_MIN_GAP_M = 45.0 * K    # ...and their clearance from any other junction
CROSS_MIN_M = 50.0 * K        # anything shorter than this is a driveway, not a street


def main():
    print("=== Generating OSM data for the English map ===")
    rng = random.Random(SEED)
    s = ms.GRID_S
    n = ms.GRID_N

    # ------------------------------------------------------------------ 0. sources
    print("0. Reading the inspiration imagery...")
    wood_src, _, density = ms.load_visual()
    wood_grid = ms.to_grid(wood_src)
    river = ms.load_river_path()
    pads, village = ms.settlement_pads()
    print(f"   woods {wood_grid.mean()*100:.1f}% of the playable area, "
          f"{len(pads)} settlements (village {village['ha']:.1f} ha at "
          f"{village['cx']:.0f},{village['cy']:.0f})")

    # ------------------------------------------------------------------ node pools
    nodes = {}
    node_coords = {}
    next_node_id = [1]

    def get_node(x, y):
        key = (round(float(x), 3), round(float(y), 3))
        if key not in nodes:
            lat, lon = ms.local_to_global(*key)
            nodes[key] = next_node_id[0]
            node_coords[next_node_id[0]] = (lat, lon)
            next_node_id[0] += 1
        return nodes[key]

    ways = []
    next_way_id = [1]

    def add_way(coords, tags):
        pts = [(float(x), float(y)) for x, y in coords]
        ways.append({'id': next_way_id[0],
                     'node_refs': [get_node(x, y) for x, y in pts],
                     'coords': pts,
                     'tags': tags})
        next_way_id[0] += 1
        return ways[-1]

    # ------------------------------------------------------------------ 1. water
    print("1. River and lake...")
    lake_grid = ms.lake_mask(n, s)

    river_img = Image.new("L", (n, n), 0)
    ImageDraw.Draw(river_img).line([(x / s, y / s) for x, y in river], fill=255,
                                   width=max(1, int(round(2 * ms.RIVER_HALF_M / s))),
                                   joint="curve")
    # The channel stops at the shore: the lake is its own polygon, and two overlapping
    # water areas would fight over the same ground in the editor.
    river_mask = (np.array(river_img) > 0) & ~lake_grid
    water_mask = river_mask | lake_grid

    lake = ms.trace_components(lake_grid, s, 1.0, 6.0)
    lake.sort(key=lambda z: -z[1])
    for i, (ring, ha) in enumerate(lake, 1):
        add_way(ring, {'natural': 'water', 'water': 'lake',
                       'name': 'Lake' if i == 1 else f'Lake {i}'})

    reaches = ms.trace_components(river_mask, s, 0.3 * K * K, 4.0)
    # Name the reaches upstream to downstream, i.e. by how far along the river they sit.
    reaches.sort(key=lambda z: centroid(z[0])[0] + centroid(z[0])[1])
    for i, (ring, ha) in enumerate(reaches, 1):
        name = 'River' if len(reaches) == 1 else \
            ('River (above the lake)' if i == 1 else
             'River (below the lake)' if i == len(reaches) else f'River (reach {i})')
        add_way(ring, {'natural': 'water', 'waterway': 'riverbank', 'name': name})
    print(f"   lake {sum(h for _, h in lake):.1f} ha at {ms.lake_surface_z():.2f} m, "
          f"river {len(reaches)} reach(es), {sum(h for _, h in reaches):.1f} ha")

    # ------------------------------------------------------------------ 2. woodland
    print("2. Woodland...")
    # The riparian tree belts in the source photo overhang the water, and a couple of
    # copses sit inside a settlement. Realistic from the air, but in game it would put
    # forest ground on top of the river and inside a yard, so both are cut away first.
    pad_img = Image.new("L", (n, n), 0)
    pdraw = ImageDraw.Draw(pad_img)
    for pad in pads:
        pdraw.polygon([(x / s, y / s) for x, y in ms.grow_ring(pad["ring"],
                       WOOD_SIMPLIFY_M)], fill=255, outline=255)
    keep_clear = ndimage.binary_dilation(water_mask | (np.array(pad_img) > 0),
                                         ms.disk(WOOD_SIMPLIFY_M / s))
    wood_grid &= ~keep_clear
    woods = ms.trace_components(wood_grid, s, MIN_WOOD_HA, WOOD_SIMPLIFY_M)
    woods.sort(key=lambda z: -z[1])
    for i, (ring, ha) in enumerate(woods, 1):
        add_way(ring, {'natural': 'wood', 'landuse': 'farmyard',
                       'leaf_type': 'broadleaved', 'name': f'Wood {i} ({ha:.1f} ha)'})
    print(f"   {len(woods)} woods, {sum(h for _, h in woods):.0f} ha")

    # ------------------------------------------------------------------ 3. farmyards
    print("3. Village and industrial pads...")
    pad_index = 0
    for pad in pads:
        if pad["village"]:
            add_way(pad["ring"], {'landuse': 'farmyard',
                                  'name': f'Village ({pad["ha"]:.1f} ha)'})
        else:
            pad_index += 1
            pad["name"] = f'Industry Pad {pad_index} ({pad["ha"]:.1f} ha)'
            add_way(pad["ring"], {'landuse': 'farmyard', 'name': pad["name"]})
    print(f"   village + {pad_index} industrial pads, "
          f"{sum(p['ha'] for p in pads):.0f} ha of flat farmyard")

    # ------------------------------------------------------------------ 4. field seeds
    print("4. Field seeds from the hedgerow density...")
    lo, hi = np.percentile(density, 8), np.percentile(density, 92)
    t = np.clip((density - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    spacing = (SPACING_OPEN_M + (SPACING_DENSE_M - SPACING_OPEN_M) * t) * SPACING_SCALE
    seeds = poisson_variable(spacing, ms.PLAYABLE_M, ms.PLAYABLE_M, rng)
    print(f"   {len(seeds)} seeds, spacing {spacing.min():.0f}-{spacing.max():.0f} m "
          f"({ms.PLAYABLE_M**2/len(seeds)/10000:.1f} ha per cell on average)")

    # A ring of distant ghost points keeps every real cell bounded, so no Voronoi region
    # runs off to infinity and needs special-casing.
    ghost = [(ms.PLAYABLE_M / 2 + 4 * ms.PLAYABLE_M * math.cos(a),
              ms.PLAYABLE_M / 2 + 4 * ms.PLAYABLE_M * math.sin(a))
             for a in np.linspace(0, 2 * math.pi, 64, endpoint=False)]
    vor = Voronoi(np.vstack([seeds, ghost]))
    verts = vor.vertices

    # ------------------------------------------------------------------ 5. lanes
    print("5. Lane network over the Voronoi edges...")
    lane_chains, lane_rank = build_lanes(vor, verts, seeds, river, wood_grid, pads,
                                         village, s, lake_grid)
    rank_tag = {0: 'primary', 1: 'secondary', 2: 'tertiary'}
    rank_name = {0: 'Main Road', 1: 'Village Road', 2: None}

    lane_ways = []
    river_tree = cKDTree(river)
    lane_chains = prune_stubs(lane_chains, STUB_MIN_M)
    before = len(lane_chains)
    lane_chains, gates = clip_at_village(lane_chains, village)
    print(f"   {len(gates)} village entrance(s); {before} lane(s) -> "
          f"{len(lane_chains)} after trimming them at the boundary")
    for chain, rank in lane_chains:
        for piece, is_bridge in split_at_river(chain, river_tree):
            tags = {'highway': rank_tag[rank]}
            if rank_name[rank]:
                tags['name'] = rank_name[rank]
            if is_bridge:
                tags['bridge'] = 'yes'
                tags['layer'] = '1'
            lane_ways.append(add_way(piece, tags))
    n_bridge = sum(1 for w in lane_ways if 'bridge' in w['tags'])
    total_km = sum(ms.polyline_length(w['coords']) for w in lane_ways) / 1000.0
    print(f"   {len(lane_ways)} ways, {total_km:.1f} km of road, {n_bridge} bridge(s)")

    # ------------------------------------------------------------------ 5b. access
    print("5b. Farmyard access...")
    n_spur = connect_pads(pads, lane_ways, add_way, get_node, lake_grid)
    print(f"   {n_spur} access spur(s) added")

    # ------------------------------------------------------------------ 6. streets
    print("6. Village streets...")
    n_streets = village_streets(village, gates, add_way, river_tree)
    print(f"   {n_streets} street(s) inside the village farmyard")

    # ------------------------------------------------------------------ 7. fields
    print("7. Cutting the fields...")
    blocked = build_blocked_mask(n, s, woods, pads, river, ways, lake_grid)
    print(f"   {blocked.mean()*100:.1f}% of the playable area is already occupied")
    fields = cut_fields(seeds, blocked, n, s)

    if len(fields) > MAX_FIELDS:
        fields.sort(key=lambda z: -z[1])
        dropped = len(fields) - MAX_FIELDS
        fields = fields[:MAX_FIELDS]
        print(f"   dropped the {dropped} smallest parcels to stay under "
              f"{MAX_FIELDS} fields")
    # Number them north to south, so the in-game field list reads like the map.
    fields.sort(key=lambda z: (round(centroid(z[0])[1] / (250.0 * K)), centroid(z[0])[0]))
    for i, (ring, ha) in enumerate(fields, 1):
        add_way(ring, {'landuse': 'farmland', 'name': f'Field {i} ({ha:.1f} ha)'})
    areas = np.array([h for _, h in fields])
    print(f"   {len(fields)} fields: {areas.min():.1f} / {np.median(areas):.1f} / "
          f"{areas.max():.1f} ha (min/median/max), {areas.sum():.0f} ha farmed")

    # ------------------------------------------------------------------ 8. junctions
    print("8. Splicing shared nodes at road crossings...")
    n_junctions = connect_road_crossings(ways, get_node)
    print(f"   {n_junctions} node(s) spliced")

    # ------------------------------------------------------------------ 9. write
    write_osm(ways, node_coords)


# ====================================================================== helpers
def centroid(ring):
    return (sum(p[0] for p in ring[:-1]) / (len(ring) - 1),
            sum(p[1] for p in ring[:-1]) / (len(ring) - 1))


def poisson_variable(spacing_field, w, h, rng, k_tries=24):
    """Bridson's Poisson-disk sampling with a per-location minimum spacing."""
    r_min = float(spacing_field.min())
    cell = r_min / math.sqrt(2.0)
    gw, gh = int(w / cell) + 1, int(h / cell) + 1
    grid = -np.ones((gh, gw), dtype=np.int32)
    pts, active = [], []
    fh, fw = spacing_field.shape
    step_y, step_x = h / fh, w / fw

    def rad_at(x, y):
        return float(spacing_field[min(max(int(y / step_y), 0), fh - 1),
                                   min(max(int(x / step_x), 0), fw - 1)])

    span = int(math.ceil(float(spacing_field.max()) / cell)) + 1

    def free(x, y):
        if not (0 <= x < w and 0 <= y < h):
            return False
        r = rad_at(x, y)
        gi, gj = int(y / cell), int(x / cell)
        for a in range(max(0, gi - span), min(gh, gi + span + 1)):
            for b in range(max(0, gj - span), min(gw, gj + span + 1)):
                q = grid[a, b]
                if q >= 0:
                    px, py = pts[q]
                    if math.hypot(px - x, py - y) < max(r, rad_at(px, py)):
                        return False
        return True

    def add(x, y):
        pts.append((x, y))
        grid[int(y / cell), int(x / cell)] = len(pts) - 1
        active.append(len(pts) - 1)

    add(w * 0.5, h * 0.5)
    while active:
        idx = active[rng.randrange(len(active))]
        px, py = pts[idx]
        r = rad_at(px, py)
        for _ in range(k_tries):
            ang = rng.uniform(0.0, 2 * math.pi)
            dist = rng.uniform(r, 2 * r)
            x, y = px + dist * math.cos(ang), py + dist * math.sin(ang)
            if free(x, y):
                add(x, y)
                break
        else:
            active.remove(idx)
    return np.array(pts)


def build_lanes(vor, verts, seeds, river, wood_grid, pads, village, s, lake_grid):
    """Pick the lanes out of the Voronoi edge graph.

    Every edge is weighted by its length plus a penalty for crossing the river or a wood,
    then a shortest-path tree is grown from the village: first out to each settlement and
    to the map edges, then on to whatever corner of the map is still unserved. Edges get
    the rank of the most important path that used them, which is what gives the network a
    hierarchy instead of a uniform mesh.
    """
    margin = 150.0 * K
    inside = ((verts[:, 0] > -margin) & (verts[:, 0] < ms.PLAYABLE_M + margin) &
              (verts[:, 1] > -margin) & (verts[:, 1] < ms.PLAYABLE_M + margin))
    river_tree = cKDTree(river)
    wood_pts = np.argwhere(wood_grid)[:, ::-1] * s + s / 2.0
    wood_tree = cKDTree(wood_pts) if len(wood_pts) else None

    adj = {}
    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0 or not (inside[a] and inside[b]):
            continue
        length = float(np.hypot(*(verts[a] - verts[b])))
        if length < 1e-6:
            continue
        mid = (verts[a] + verts[b]) / 2.0
        # A bridge spans a river; nothing spans a lake, so those edges are simply not
        # in the graph and the network has to go round.
        if segment_in_lake(verts[a], verts[b], lake_grid, s):
            continue
        weight = length
        if river_tree.query(mid)[0] < ms.RIVER_HALF_M + 25.0:
            weight += RIVER_CROSS_PENALTY
        if wood_tree is not None and wood_tree.query(mid)[0] < 25.0 * K:
            weight += WOOD_CROSS_PENALTY
        adj.setdefault(a, []).append((b, weight))
        adj.setdefault(b, []).append((a, weight))

    def dijkstra(sources):
        dist = {v: 0.0 for v in sources}
        parent = {}
        queue = [(0.0, v) for v in sources]
        heapq.heapify(queue)
        while queue:
            d, v = heapq.heappop(queue)
            if d > dist.get(v, math.inf):
                continue
            for u, w in adj.get(v, ()):
                nd = d + w
                if nd < dist.get(u, math.inf):
                    dist[u] = nd
                    parent[u] = v
                    heapq.heappush(queue, (nd, u))
        return dist, parent

    # Only vertices that are actually in the graph can be linked to. A vertex stranded
    # inside the lake, or one every edge of which was dropped, would be picked as the
    # nearest to some seed and then never reached.
    usable = inside & np.array([i in adj for i in range(len(verts))])
    if not usable.any():
        raise SystemExit("No usable Voronoi vertices for the lane network.")
    idx_inside = np.nonzero(usable)[0]
    vtree = cKDTree(verts[usable])

    def nearest(pt):
        return int(idx_inside[vtree.query(pt)[1]])

    start = nearest((village["cx"], village["cy"]))
    tree = {start}
    rank_of = {}

    def link(target, rank):
        dist, parent = dijkstra(list(tree))
        if target not in dist:
            return False
        v = target
        while v in parent:
            key = frozenset((v, parent[v]))
            rank_of[key] = min(rank_of.get(key, 9), rank)
            tree.add(v)
            v = parent[v]
        tree.add(v)
        return True

    # rank 0: the through route, west edge -> village -> east edge
    link(nearest((4.0, ms.PLAYABLE_M * 0.45)), 0)
    link(nearest((ms.PLAYABLE_M - 4.0, ms.PLAYABLE_M * 0.55)), 0)
    # rank 1: village out to each settlement, and to the remaining map edges
    for pad in pads:
        if not pad["village"]:
            link(nearest((pad["cx"], pad["cy"])), 1)
    for pt in [(ms.PLAYABLE_M * 0.5, 4.0), (ms.PLAYABLE_M * 0.5, ms.PLAYABLE_M - 4.0),
               (ms.PLAYABLE_M * 0.18, 4.0),
               (ms.PLAYABLE_M * 0.82, ms.PLAYABLE_M - 4.0)]:
        link(nearest(pt), 1)
    # rank 2: farm lanes, until no field seed is stranded.
    # Seeds already tried are struck off: without that, one seed the network genuinely
    # cannot reach - a cell drowned by the lake, say - is picked again on every pass,
    # and the loop burns its whole budget without laying another metre of lane.
    tried = np.zeros(len(seeds), dtype=bool)
    drowned = np.array([in_lake(pt, lake_grid, s, margin_px=1) for pt in seeds])
    tried |= drowned
    for _ in range(120):
        served = np.array([verts[i] for i in tree])
        far = cKDTree(served).query(seeds)[0]
        far[tried] = -1.0
        k = int(np.argmax(far))
        if far[k] < LANE_COVERAGE_M:
            break
        tried[k] = True
        link(nearest(seeds[k]), 2)
    if drowned.any():
        print(f"   {int(drowned.sum())} field seed(s) under the lake, not served")

    # ---- chain the chosen edges into polylines, splitting at junctions
    graph = {}
    for key in rank_of:
        a, b = tuple(key)
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)

    used = set()
    chains = []

    def walk(a, b):
        chain = [a, b]
        used.add(frozenset((a, b)))
        while True:
            cur, prev = chain[-1], chain[-2]
            if len(graph[cur]) != 2:
                break
            nxt = [v for v in graph[cur] if v != prev]
            if not nxt or frozenset((cur, nxt[0])) in used:
                break
            used.add(frozenset((cur, nxt[0])))
            chain.append(nxt[0])
        return chain

    for a in sorted(graph):
        if len(graph[a]) == 2:
            continue
        for b in graph[a]:
            if frozenset((a, b)) not in used:
                chains.append(walk(a, b))
    # anything left is a closed loop of degree-2 vertices
    for key in sorted(rank_of, key=lambda k: sorted(k)):
        if key not in used:
            a, b = tuple(key)
            chains.append(walk(a, b))

    out = []
    for chain in chains:
        rank = min(rank_of[frozenset((chain[i], chain[i + 1]))]
                   for i in range(len(chain) - 1))
        pts = [tuple(verts[i]) for i in chain]
        # Corner cutting keeps the endpoints, so the junctions stay shared nodes while
        # the run between them stops looking like a Voronoi diagram.
        pts = ms.chaikin(pts, LANE_SMOOTH_ITERS)
        pts = [(min(max(x, 0.0), ms.PLAYABLE_M), min(max(y, 0.0), ms.PLAYABLE_M))
               for x, y in pts]
        pts = ms.simplify(pts, 2.0)
        if len(pts) >= 2 and ms.polyline_length(pts) > 25.0 * ms.MAP_SCALE:
            out.append((pts, rank))
    return out, rank_of


def in_lake(pt, lake_grid, s, margin_px=2):
    """Is this point in (or just beside) the lake, on the working raster?"""
    n = lake_grid.shape[0]
    j = int(pt[0] / s)
    i = int(pt[1] / s)
    if not (0 <= i < n and 0 <= j < n):
        return False
    i0, i1 = max(0, i - margin_px), min(n, i + margin_px + 1)
    j0, j1 = max(0, j - margin_px), min(n, j + margin_px + 1)
    return bool(lake_grid[i0:i1, j0:j1].any())


def segment_in_lake(a, b, lake_grid, s, step=20.0):
    """Does any part of this segment touch the lake?

    The midpoint alone is not enough: a Voronoi edge is hundreds of metres long at this
    scale and can clip a lobe of the lake while its centre sits comfortably on dry land.
    """
    steps = max(2, int(math.dist(a, b) / step) + 1)
    for i in range(steps + 1):
        t = i / steps
        if in_lake((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])), lake_grid, s):
            return True
    return False


def point_in_ring(pt, ring):
    inside = False
    pts = ring[:-1]
    for i, j in zip(range(len(pts)), range(-1, len(pts) - 1)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > pt[1]) != (yj > pt[1]) and \
                pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi:
            inside = not inside
    return inside


def connect_pads(pads, lane_ways, add_way, get_node, lake_grid=None):
    """Run a spur from the nearest lane into every farmyard that has no road at its gate.

    The lane tree is grown to the Voronoi vertex closest to each settlement, which lands
    near the pad but rarely on it. Without this the industrial pads are reachable only
    across a field, which is exactly the kind of thing you only notice once the map is
    in game.
    """
    added = 0
    for pad in pads:
        ring = pad["ring"]
        cx, cy = pad["cx"], pad["cy"]

        # already served? a lane vertex inside the pad, or within a few metres of it
        best = None
        for w in lane_ways:
            for k, pt in enumerate(w['coords']):
                if point_in_ring(pt, ring):
                    best = None
                    break
                d = math.hypot(pt[0] - cx, pt[1] - cy)
                if best is None or d < best[0]:
                    best = (d, w, k, pt)
            else:
                continue
            break
        if best is None or best[0] > PAD_LINK_MAX_M:
            continue

        _, way, k, pt = best
        # aim at the ring vertex closest to the lane, so the spur meets the pad edge
        gate = min(ring[:-1], key=lambda q: math.hypot(q[0] - pt[0], q[1] - pt[1]))
        # ...and push a little past it, so the spur ends inside the farmyard
        dx, dy = cx - gate[0], cy - gate[1]
        dl = math.hypot(dx, dy)
        if dl > 1e-6:
            gate = (gate[0] + dx / dl * 12.0 * K, gate[1] + dy / dl * 12.0 * K)
        if math.dist(pt, gate) < 15.0 * K:
            continue
        if lake_grid is not None and segment_in_lake(pt, gate, lake_grid, ms.GRID_S):
            continue
        add_way([pt, gate], {'highway': 'tertiary', 'name': 'Farm Access'})
        # `pt` is an existing vertex of `way`, so get_node hands both ways the same id
        added += 1
    return added


def prune_stubs(chains, min_len):
    """Drop the short dead-end lanes.

    The coverage pass links whichever field seed is furthest from the network, and the
    last few metres of such a path can leave a stub that starts at a junction and ends
    in the middle of a field. A farm track that actually goes somewhere is kept - dead
    ends are normal on English lanes - but the offcuts are not.
    """
    ends = {}
    for pts, _ in chains:
        for p in (pts[0], pts[-1]):
            key = (round(p[0], 2), round(p[1], 2))
            ends[key] = ends.get(key, 0) + 1

    def at_edge(p):
        return (p[0] < 25.0 or p[1] < 25.0 or p[0] > ms.PLAYABLE_M - 25.0
                or p[1] > ms.PLAYABLE_M - 25.0)

    kept = []
    for pts, rank in chains:
        leaves = sum(1 for p in (pts[0], pts[-1])
                     if ends[(round(p[0], 2), round(p[1], 2))] < 2 and not at_edge(p))
        if leaves and ms.polyline_length(pts) < min_len:
            continue
        kept.append((pts, rank))
    if len(kept) != len(chains):
        print(f"   pruned {len(chains) - len(kept)} dead-end stub(s) under "
              f"{min_len:.0f} m")
    return kept


def split_at_river(chain, river_tree):
    """Split a lane where it crosses the water, so the crossing can be tagged a bridge.

    The test runs on a densified copy: after the corner cutting and the Douglas-Peucker
    pass a lane can step clean over a 22 m river between two vertices, and the crossing
    would go untagged. Each piece is simplified again afterwards, so only the bridge
    itself keeps the extra nodes.
    """
    if ms.polyline_length(chain) < 12.0:
        return [(chain, False)]
    dense = [tuple(p) for p in ms.resample(chain, 6.0)]
    if math.dist(dense[-1], chain[-1]) > 1e-6:
        dense.append(tuple(chain[-1]))
    near = river_tree.query(np.array(dense))[0] <= BRIDGE_HALF_M
    if not near.any():
        return [(chain, False)]

    pieces = []
    start = 0
    for i in range(1, len(dense)):
        if near[i] != near[i - 1]:
            piece = dense[start:i + 1]
            if len(piece) >= 2:
                pieces.append((piece, bool(near[i - 1])))
            start = i
    tail = dense[start:]
    if len(tail) >= 2:
        pieces.append((tail, bool(near[-1])))
    if not pieces:
        return [(chain, False)]
    return [(ms.simplify(pts, 1.0 if is_bridge else 2.5), is_bridge)
            for pts, is_bridge in pieces]


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def clip_at_village(chains, village):
    """Cut every lane at the village boundary and note where it arrived.

    The lanes are chosen from the Voronoi edges, which know nothing about the village:
    half a dozen of them wander straight across it, duplicating the streets laid on top
    and crossing them at whatever angle they happened to have. Trimming them at the ring
    turns that into a set of entrances, and what goes inside is then a layout rather
    than the residue of two networks overlaid.
    """
    ring = village["ring"]

    def crossing(a, b):
        """Where the segment a->b passes through the ring, by bisection."""
        a_in = point_in_ring(a, ring)
        lo, hi = 0.0, 1.0
        for _ in range(32):
            mid = (lo + hi) / 2.0
            if point_in_ring(lerp(a, b, mid), ring) == a_in:
                lo = mid
            else:
                hi = mid
        return lerp(a, b, (lo + hi) / 2.0)

    kept, gates = [], []
    for pts, rank in chains:
        ins = [point_in_ring(p, ring) for p in pts]
        if not any(ins):
            kept.append((pts, rank))
            continue
        if all(ins):
            continue                       # wholly inside: the streets replace it
        run = []
        for i, p in enumerate(pts):
            if ins[i]:
                if run:                    # ...leaving the fields, entering the village
                    g = crossing(run[-1], p)
                    gates.append((g, rank))
                    kept.append((run + [g], rank))
                    run = []
                continue
            if i and ins[i - 1]:           # ...coming back out of it
                g = crossing(pts[i - 1], p)
                gates.append((g, rank))
                run.append(g)
            run.append(p)
        if len(run) >= 2:
            kept.append((run, rank))

    # Two lanes arriving a few metres apart are one entrance, not two: left as they are
    # they give the village a pair of near-parallel streets running to the same place.
    # The later arrival is pulled onto the first, and any vertex it had inside the merge
    # radius is dropped so it swings in cleanly rather than kinking at the last metre.
    reps = []
    snap = {}
    for pt, rank in gates:
        for r in reps:
            if math.dist(pt, r["pt"]) < GATE_MERGE_M:
                r["rank"] = min(r["rank"], rank)
                break
        else:
            r = {"pt": pt, "rank": rank}
            reps.append(r)
        snap[(round(pt[0], 3), round(pt[1], 3))] = r

    out = []
    for pts, rank in kept:
        pts = list(pts)
        for end in (0, -1):
            r = snap.get((round(pts[end][0], 3), round(pts[end][1], 3)))
            if r is None or r["pt"] == pts[end]:
                continue
            body = pts[1:] if end == 0 else pts[:-1]
            body = [p for p in body if math.dist(p, r["pt"]) > GATE_MERGE_M]
            pts = [r["pt"]] + body if end == 0 else body + [r["pt"]]
        if len(pts) >= 2 and ms.polyline_length(pts) > 30.0:
            out.append((pts, rank))
    return out, reps


def polyline_at(poly, t):
    """The point a given distance along a polyline, with the segment it landed on."""
    acc = 0.0
    for i in range(len(poly) - 1):
        seg = math.dist(poly[i], poly[i + 1])
        if seg < 1e-9:
            continue
        if acc + seg >= t:
            u = (t - acc) / seg
            return lerp(poly[i], poly[i + 1], u), i, u
        acc += seg
    return poly[-1], max(0, len(poly) - 2), 1.0


def weave(poly, extra):
    """Rebuild a polyline through points that already lie on it.

    `extra` maps a segment index to the (fraction along it, point) pairs landing there.
    Putting them in as vertices is what makes a junction a shared node: two ways that
    merely touch at a coordinate one of them does not carry are not joined at all.
    """
    out = [poly[0]]
    for i in range(len(poly) - 1):
        for _, q in sorted(extra.get(i, [])):
            if math.dist(q, out[-1]) > 1e-6:
                out.append(q)
        if math.dist(poly[i + 1], out[-1]) > 1e-6:
            out.append(poly[i + 1])
    return out


def ray_hit(origin, direction, poly):
    """Where a ray leaving `origin` first meets the polyline `poly`.

    Returns the point together with the segment it landed on, so the polyline can be
    rebuilt through it rather than left hoping the crossing pass notices them.
    """
    ox, oy = origin
    dx, dy = direction
    best = None
    for i in range(len(poly) - 1):
        ax, ay = poly[i]
        sx, sy = poly[i + 1][0] - ax, poly[i + 1][1] - ay
        den = dx * sy - dy * sx
        if abs(den) < 1e-12:
            continue
        t = ((ax - ox) * sy - (ay - oy) * sx) / den      # along the ray
        u = ((ax - ox) * dy - (ay - oy) * dx) / den      # along the segment
        if t > 1e-6 and -1e-9 <= u <= 1.0 + 1e-9 and (best is None or t < best[0]):
            best = (t, i, u, (ox + dx * t, oy + dy * t))
    return None if best is None else (best[3], best[1], best[2])


def village_streets(village, gates, add_way, river_tree):
    """Lay the streets inside the village farmyard.

    A high street between the two most nearly opposite entrances, a side street from
    each of the rest, a green closed by two back lanes, a perimeter lane just inside the
    boundary, and cross streets tying the three together. The brief asks for the whole
    village to be one farmyard, so this is what keeps it reading as a village from the
    air rather than as a field with a road across it.

    Every centreline is smoothed *before* anything is cast at it, and every junction is
    woven into the line it lands on as a real vertex. Aiming a street at the unsmoothed
    line and then emitting the smoothed one leaves the two a metre apart - close enough
    to look joined, far enough not to be.
    """
    ring = village["ring"]
    mid_pt = (village["cx"], village["cy"])
    if len(gates) < 2:
        return 0

    def hold_inside(p, pull=0.35):
        return p if point_in_ring(p, ring) else lerp(p, mid_pt, pull)

    def smooth(pts, iters=2):
        # The ends are left exactly where they were: an entrance sits *on* the ring, so
        # `hold_inside` cannot tell which side of it the point is, and nudging one by a
        # millimetre gives it a node id of its own - which silently cuts the street off
        # from the lane that arrives there.
        out = ms.chaikin(pts, iters)
        return [out[0]] + [hold_inside(p) for p in out[1:-1]] + [out[-1]]

    def emit(pts, tags):
        for piece, is_bridge in split_at_river(pts, river_tree):
            t = dict(tags)
            if is_bridge:
                t['bridge'] = 'yes'
                t['layer'] = '1'
            add_way(piece, t)

    # --- the high street: the pair of entrances facing each other most squarely, and
    # among equally opposed pairs the one the through route already uses.
    best = None
    for i in range(len(gates)):
        for j in range(i + 1, len(gates)):
            ga, gb = gates[i], gates[j]
            va = (ga["pt"][0] - mid_pt[0], ga["pt"][1] - mid_pt[1])
            vb = (gb["pt"][0] - mid_pt[0], gb["pt"][1] - mid_pt[1])
            cos = ((va[0] * vb[0] + va[1] * vb[1])
                   / (math.hypot(*va) * math.hypot(*vb) + 1e-9))
            score = (ga["rank"] + gb["rank"], cos)     # both lower is better
            if best is None or score < best[0]:
                best = (score, ga, gb)
    head, tail = best[1], best[2]

    # A gentle S laid over a pull towards the middle of the pad: bending the control
    # points straight at the centre does nothing when the two entrances already face
    # each other through it, and the street comes out ruler-straight.
    a, b = head["pt"], tail["pt"]
    chord = math.dist(a, b) or 1.0
    ux, uy = (b[0] - a[0]) / chord, (b[1] - a[1]) / chord
    nx, ny = -uy, ux
    waist = lerp(a, b, 0.5)
    pull = (mid_pt[0] - waist[0]) * nx + (mid_pt[1] - waist[1]) * ny
    ctrl = []
    for f, sign in ((0.30, 1.0), (0.70, -1.0)):
        q = lerp(a, b, f)
        d = sign * chord * 0.07 + pull * 0.6
        ctrl.append(hold_inside((q[0] + nx * d, q[1] + ny * d)))
    spine = smooth([a] + ctrl + [b], 3)
    total = ms.polyline_length(spine)

    def at(t):
        return polyline_at(spine, min(max(t, 0.0), total))

    def normal(t):
        p = at(max(0.0, t - 12.0))[0]
        q = at(min(total, t + 12.0))[0]
        dx, dy = q[0] - p[0], q[1] - p[1]
        d = math.hypot(dx, dy) or 1.0
        return (-dy / d, dx / d)

    on_spine = {}          # junctions to weave into the high street before it is written
    ways_out = []          # (points, tags) queued so the spine can be written last

    def join_spine(t):
        p, seg, u = at(t)
        on_spine.setdefault(seg, []).append((u, p))
        return p

    # --- the perimeter lane, just inside the boundary. It goes first because
    # everything else has to stand clear of where the high street runs through it.
    peri = [tuple(p) for p in ms.chaikin(ms.grow_ring(ring, -STREET_INSET_M)[:-1],
                                         2, closed=True)]
    peri.append(peri[0])

    # --- the green is tied in next, so the side streets do not land on its junctions.
    # Its ends are pulled clear of where the high street passes through the perimeter:
    # at a fixed fraction of the spine they land a few tens of metres from that
    # crossing - the perimeter is a fixed inset, so the two track each other - and the
    # pair reads as one knot however well spaced everything else is.
    lo, hi = total * 0.22, total * 0.78
    within = [u for u in np.arange(0.0, total, 5.0)
              if point_in_ring(polyline_at(spine, u)[0], peri)]
    if within and within[-1] - within[0] > 4.0 * CROSS_MIN_GAP_M:
        lo = max(lo, within[0] + CROSS_MIN_GAP_M)
        hi = min(hi, within[-1] - CROSS_MIN_GAP_M)
    # The two crossings count as junctions in their own right, so a side street keeps
    # its distance from them as well.
    taken = [0.0, total, lo, hi] + ([within[0], within[-1]] if within else [])

    # --- a side street per remaining entrance, each at its own T-junction
    for g in gates:
        if g is head or g is tail:
            continue
        ideal = float(min(range(int(total) + 1),
                          key=lambda u: math.dist(at(float(u))[0], g["pt"])))
        # Pick from the positions that actually clear every junction already on the
        # street. Nudging the ideal away from the nearest one and clamping it back into
        # range - which is what this did - lets the clamp win: the junction ends up
        # pinned at the end of the range a few metres from the one it was moved to
        # avoid, and the two read as one knot.
        edge = STREET_MIN_GAP_M * 0.6
        free = [u for u in np.arange(edge, max(edge + 1.0, total - edge), 5.0)
                if min(abs(u - v) for v in taken) >= STREET_MIN_GAP_M]
        if free:
            t = min(free, key=lambda u: abs(u - ideal))
        else:
            # nothing clears it: take the middle of the widest gap left, so the
            # junctions end up as far apart as the street allows
            marks = sorted(taken)
            t = max(((marks[i] + marks[i + 1]) / 2.0, marks[i + 1] - marks[i])
                    for i in range(len(marks) - 1))[0]
        taken.append(t)
        junction = join_spine(t)
        jx, jy = normal(t)
        bow = math.dist(g["pt"], junction) * 0.10
        mid = lerp(g["pt"], junction, 0.5)
        ways_out.append((smooth([g["pt"], (mid[0] + jx * bow, mid[1] + jy * bow),
                                 junction]),
                         {'highway': 'tertiary', 'name': 'Village Street'}))

    # --- the green: a lens either side of the high street. Both sides usually have the
    # room, and taking both leaves the village a block either side of its main street
    # rather than one built-up flank and one bare one.
    ends = (join_spine(lo), join_spine(hi))
    gx, gy = normal((lo + hi) / 2.0)
    waists = [at(lo + (hi - lo) * f)[0] for f in (0.25, 0.5, 0.75)]
    arcs = []
    for sign in (1.0, -1.0):
        # The green takes half the room on its side, not a fixed width: the pad is not
        # symmetrical about its high street, and a fixed offset either overruns the
        # perimeter on the narrow flank - leaving no block to put a street in - or
        # leaves the wide one nearly empty.
        room = []
        for q in waists:
            hit = ray_hit(q, (gx * sign, gy * sign), peri)
            room.append(math.inf if hit is None else math.dist(q, hit[0]))
        off = min(BACK_LANE_OFF_M, min(room) * 0.5)
        if off < 40.0 * K:            # no room on this flank for a green at all
            continue
        arc = [ends[0]]
        for q, scale in zip(waists, (0.8, 1.0, 0.8)):
            d = sign * off * scale
            arc.append((q[0] + gx * d, q[1] + gy * d))
        arc.append(ends[1])
        if all(point_in_ring(p, ring) for p in arc[1:-1]):
            arcs.append((sign, smooth(arc, 2)))

    # --- cross streets: walk the perimeter and run one inward from each station to
    # the first thing it meets - the green where there is one on that flank, the high
    # street otherwise. Casting from the perimeter rather than from the high street is
    # --- cross streets running from the green out to the perimeter, so the block
    # between the two is served. They leave the green square to it - straight out from
    # the nearest point of the high street - rather than all aiming at one place, which
    # is how the fan of spokes this layout replaced came about.
    on_peri, on_arc = {}, [{} for _ in arcs]
    crossings = []
    # Everything drawn so far, so a cross street can be kept off it. The ends of the
    # green already sit on the high street beside a side street junction, and one more
    # landing next to those turns legible junctions into a knot.
    laid = cKDTree(np.vstack([ms.resample(pts, 12.0) for pts, _ in ways_out])) \
        if ways_out else None
    for k, (_, arc) in enumerate(arcs):
        arc_len = ms.polyline_length(arc)
        t = arc_len * 0.28          # ...and off the ends of the green itself
        while t <= arc_len * 0.72:
            p, seg, u = polyline_at(arc, t)
            t += CROSS_SPACING_M
            near = min(spine, key=lambda q: math.dist(q, p))
            dx, dy = p[0] - near[0], p[1] - near[1]
            dl = math.hypot(dx, dy)
            if dl < 1e-6:
                continue
            hit = ray_hit(p, (dx / dl, dy / dl), peri)
            if hit is None or math.dist(p, hit[0]) < CROSS_MIN_M:
                continue
            if laid is not None and laid.query(hit[0])[0] < CROSS_MIN_GAP_M:
                continue
            if any(math.dist(hit[0], q) < CROSS_MIN_GAP_M or
                   math.dist(p, o) < CROSS_MIN_GAP_M for o, q in crossings):
                continue
            on_arc[k].setdefault(seg, []).append((u, p))
            on_peri.setdefault(hit[1], []).append((hit[2], hit[0]))
            crossings.append((p, hit[0]))

    for p, q in crossings:
        mid = lerp(p, q, 0.5)
        bx, by = -(q[1] - p[1]), q[0] - p[0]
        bl = math.hypot(bx, by) or 1.0
        bow = math.dist(p, q) * 0.06
        ways_out.append((smooth([p, (mid[0] + bx / bl * bow, mid[1] + by / bl * bow), q],
                                2),
                         {'highway': 'tertiary', 'name': 'Village Street'}))

    for k, (_, arc) in enumerate(arcs):
        ways_out.append((weave(arc, on_arc[k]),
                         {'highway': 'tertiary', 'name': 'Back Lane'}))
    ways_out.append((weave(peri, on_peri),
                     {'highway': 'tertiary', 'name': 'Village Lane'}))

    rank_tag = {0: 'primary', 1: 'secondary', 2: 'secondary'}
    ways_out.append((weave(spine, on_spine),
                     {'highway': rank_tag[min(head["rank"], tail["rank"])],
                      'name': 'High Street'}))
    for pts, tags in ways_out:
        emit(pts, tags)
    return len(ways_out)


def build_blocked_mask(n, s, woods, pads, river, ways, lake_grid):
    """Everything a field may not grow into: woods (with a margin), farmyards, water
    and the road corridors."""
    wood_img = Image.new("L", (n, n), 0)
    wdraw = ImageDraw.Draw(wood_img)
    for ring, _ in woods:
        wdraw.polygon([(x / s, y / s) for x, y in ring], fill=255, outline=255)
    wood = np.array(wood_img) > 0
    if wood.any():
        wood = ndimage.binary_dilation(
            wood, ms.disk((WOOD_MARGIN_M + SIMPLIFY_SLACK_M) / s))

    img = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(img)
    for pad in pads:
        draw.polygon([(x / s, y / s) for x, y in
                      ms.grow_ring(pad["ring"], HEDGE_M + SIMPLIFY_SLACK_M)],
                     fill=255, outline=255)
    draw.line([(x / s, y / s) for x, y in river], fill=255,
              width=max(1, int(round(2 * (ms.RIVER_HALF_M + 6.0 + SIMPLIFY_SLACK_M) / s))),
              joint="curve")
    for w in ways:
        if 'highway' not in w['tags']:
            continue
        pts = [(x / s, y / s) for x, y in w['coords']]
        if len(pts) >= 2:
            draw.line(pts, fill=255, width=max(1, int(round(2 * LANE_HALF_M / s))),
                      joint="curve")
    lake = ndimage.binary_dilation(lake_grid,
                                   ms.disk((HEDGE_M + SIMPLIFY_SLACK_M) / s))
    return (np.array(img) > 0) | wood | lake


def cut_fields(seeds, blocked, n, s):
    """One parcel per Voronoi cell: take the cell, remove everything blocked, pull the
    edge back by the hedgerow width, and outline whatever survives."""
    gy, gx = np.mgrid[0:n, 0:n]
    query = np.column_stack([(gx.ravel() + 0.5) * s, (gy.ravel() + 0.5) * s])
    cell_id = cKDTree(seeds).query(query, workers=-1)[1].reshape(n, n).astype(np.int32)

    free = ~blocked
    hedge_px = max(1, int(round(HEDGE_M / s)))
    kernel = ms.disk(hedge_px)
    objs = ndimage.find_objects(cell_id + 1, max_label=len(seeds))
    fields = []
    for k in range(len(seeds)):
        sl = objs[k]
        if sl is None:
            continue
        sl = (slice(max(0, sl[0].start - 4), min(n, sl[0].stop + 4)),
              slice(max(0, sl[1].start - 4), min(n, sl[1].stop + 4)))
        mask = (cell_id[sl] == k) & free[sl]
        if not mask.any():
            continue
        mask = ndimage.binary_erosion(mask, kernel)
        if not mask.any():
            continue
        for ring, ha in ms.trace_components(
                mask, s, MIN_FIELD_HA, FIELD_SIMPLIFY_M,
                offset=(sl[1].start * s, sl[0].start * s)):
            perimeter = ms.polyline_length(ring)
            # A long thin remnant pinched off against a lane is not a workable field.
            if perimeter > 0 and 2.0 * ha * 10000.0 / perimeter < MIN_FIELD_WIDTH_M:
                continue
            fields.append((ring, ha))
    return fields


def connect_road_crossings(ways, get_node):
    """Give every at-grade road crossing a shared node.

    The lane chains already share their junction vertices, but the village streets and
    the pieces either side of a bridge are emitted independently, so anything that only
    overlaps visually is stitched here.
    """
    roads = [w for w in ways if 'highway' in w['tags']]
    spliced = 0

    def seg_int(a, b, c, d):
        rx, ry = b[0] - a[0], b[1] - a[1]
        sx, sy = d[0] - c[0], d[1] - c[1]
        den = rx * sy - ry * sx
        if abs(den) < 1e-12:
            return None
        t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / den
        u = ((c[0] - a[0]) * ry - (c[1] - a[1]) * rx) / den
        if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            return (a[0] + t * rx, a[1] + t * ry), t, u
        return None

    def has_vertex(coords, p):
        return any(abs(v[0] - p[0]) < 1e-3 and abs(v[1] - p[1]) < 1e-3 for v in coords)

    boxes = []
    for w in roads:
        xs = [p[0] for p in w['coords']]
        ys = [p[1] for p in w['coords']]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))

    inserts = {id(w): [] for w in roads}
    for i in range(len(roads)):
        for j in range(i + 1, len(roads)):
            ax0, ay0, ax1, ay1 = boxes[i]
            bx0, by0, bx1, by1 = boxes[j]
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            wa, wb = roads[i], roads[j]
            for p in range(len(wa['coords']) - 1):
                for q in range(len(wb['coords']) - 1):
                    hit = seg_int(wa['coords'][p], wa['coords'][p + 1],
                                  wb['coords'][q], wb['coords'][q + 1])
                    if not hit:
                        continue
                    pt, t, u = hit
                    pt = (round(pt[0], 3), round(pt[1], 3))
                    if not has_vertex(wa['coords'], pt):
                        inserts[id(wa)].append((p, t, pt))
                    if not has_vertex(wb['coords'], pt):
                        inserts[id(wb)].append((q, u, pt))

    for w in roads:
        todo = inserts[id(w)]
        if not todo:
            continue
        # Deepest segment first, so the earlier insertion indices stay valid.
        for seg_i, frac, pt in sorted(set(todo), key=lambda z: (-z[0], -z[1])):
            if has_vertex(w['coords'], pt):
                continue
            w['coords'].insert(seg_i + 1, pt)
            w['node_refs'].insert(seg_i + 1, get_node(*pt))
            spliced += 1
    return spliced


def write_osm(ways, node_coords):
    minlat, minlon = ms.local_to_global(0.0, ms.PLAYABLE_M)
    maxlat, maxlon = ms.local_to_global(ms.PLAYABLE_M, 0.0)

    osm = ET.Element('osm', version='0.6', generator='FS25 map pipeline')
    ET.SubElement(osm, 'bounds', {
        'minlat': f"{minlat:.10f}", 'minlon': f"{minlon:.10f}",
        'maxlat': f"{maxlat:.10f}", 'maxlon': f"{maxlon:.10f}"})

    stamp = {'version': '1', 'timestamp': '2026-09-01T12:00:00Z',
             'changeset': '1', 'uid': '1', 'user': 'generator'}
    for nid in sorted(node_coords):
        lat, lon = node_coords[nid]
        ET.SubElement(osm, 'node', {'id': str(nid), 'lat': f"{lat:.10f}",
                                    'lon': f"{lon:.10f}", **stamp})
    for way in ways:
        elem = ET.SubElement(osm, 'way', {'id': str(way['id']), **stamp})
        for ref in way['node_refs']:
            ET.SubElement(elem, 'nd', ref=str(ref))
        for k, v in way['tags'].items():
            ET.SubElement(elem, 'tag', k=k, v=v)

    pretty = minidom.parseString(ET.tostring(osm, encoding='utf-8')).toprettyxml(
        indent='  ', encoding='utf-8')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.osm")
    with open(out, "wb") as fh:
        fh.write(pretty)
    print(f"[+] Wrote {len(node_coords)} nodes and {len(ways)} ways to '{out}'.")


if __name__ == '__main__':
    main()
