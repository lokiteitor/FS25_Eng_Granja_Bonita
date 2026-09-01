# FS25 map pipeline — English river-valley map

Generates the terrain and zoning for a Farming Simulator 25 map from two reference
images, and renders them for inspection.

```
inspiracion/mapa_alturas.jpeg ─┐
inspiracion/mapa_visual.jpeg  ─┴─► map_source.py ─┬─► dem_generator ─► dem_new_12k.png
                                                  ├─► osm_generator ─► map.osm
                                pf_generator ─────┴─► visualizer ────► dem_viewer_3d.html
```

## The map

An English lowland landscape: a river meandering NW → SE through a broad valley, gentle
uplands either side, irregular hedged fields, winding lanes and scattered woodland. Part
way down its course the river opens out into a lake and leaves it again at the far end.

| | |
|---|---|
| DEM canvas | 12288 × 12288 px, **1 px = 1 m** |
| Playable area | 8192 × 8192 m, centred (offset 2048 m) |
| Heights | 16-bit **centimetres** (`raw / 100 = metres`) |
| Relief | ~22 m at the river mouth to ~110 m on the hilltops |
| Fields | ~175 parcels, 5–72 ha (median ~29 ha), ~5200 ha farmed |
| Woodland | ~22 woods, ~610 ha (9%), 6–31 corners each |
| Farmyards | 1 village (60 ha) + 11 flat industrial pads, ~310 ha |
| Water | River (~35 ha) plus a 91 ha lake on it, 6 m deep, mean 4.7 m |
| Roads | ~76 km, primary / secondary / tertiary, 4 bridges |
| Farmland slope | median 1.4°, p99 6.3°, max 16.4° |

Field sizes follow from the two constraints together: the reference image is mapped 1:1
onto the playable area and holds roughly 200 parcels, so at 8 km across they average
about 27 ha. Halving `SPACING_SCALE` in the OSM generator gives English-sized fields
(~7 ha) at around 700 parcels, well over the 200 cap.

The lake is placed by position *along the river* rather than by map coordinates, so it
stays on the water however the traced alignment moves. Across its reach the water surface
is held flat — that is what makes it a lake and not a wide bit of river — and the profile
drops again at the outlet. Nothing crosses it: lanes are routed round, and the river is
bridged only above and below it.

A wood is **not a canopy** — it is the block of ground the trees get planted on by hand
in the editor — so the outline traced off the photograph is regularised before it is
written out. Notches narrower than `WOOD_CLOSE_M` are filled, limbs thinner than
`WOOD_OPEN_M` are cut off, and specks of leaf-coloured noise go before either. What comes
out is a shape with a workable interior: a dozen or so corners rather than a hundred, no
pinched arms, no holes.

The woods are traced early, because the lanes route around them and the parcels are cut
against them, but they are **written out last**. Cutting the fields leaves crescents — a
Voronoi cell that loses most of itself to a wood comes back under `MIN_FIELD_HA` or
thinner than `MIN_FIELD_WIDTH_M`, gets dropped, and what is left is bare ground in the
shape of the wood it lies against. Those scraps go back to the wood, which is what the
photograph had growing there before the parcels were laid over it. Only the wide ones:
the gap mask is opened at `WOOD_POCKET_M` first, or the hedgerow web between the fields
is one connected component and a single wood swallows the lot.

Together that takes the ground belonging to nobody from ~9.7% of the map to ~6.5%, and
the part of it wide enough to matter that lies against a wood from ~175 ha to ~9 ha.

Only the **central village** is modelled as a settlement — one `landuse=farmyard`
covering all of its blocks, with streets on top. Every other settlement in the reference
image becomes a **flat farmyard pad** to place a production on; the DEM levels the ground
under each one to within a centimetre.

The lanes come from the Voronoi edges and know nothing about the village, so they are
trimmed at its boundary; the streets inside are then laid out from the entrances that
leaves — a high street between the two most opposite ones, a side street per remaining
entrance joining it at its own T-junction, a green closed by a back lane either side, a
perimeter lane just inside the boundary, and cross streets tying the three together.
Lanes arriving within `GATE_MERGE_M` of each other are pulled onto a single entrance.

What keeps it a layout rather than a knot is that each piece is placed against the
geometry already there, never at a guessed fraction:

- the perimeter goes down first, and the green's ends are pulled clear of where the high
  street runs **through** it — the perimeter is a fixed inset, so a green at a fixed
  fraction of the spine always lands a few tens of metres from that crossing;
- a side street takes the position nearest its own entrance that clears every junction
  already on the high street by `STREET_MIN_GAP_M`, the two crossings included;
- the green takes **half the room on its flank**, measured out to the perimeter, so it
  neither overruns the narrow side nor leaves the wide one empty;
- cross streets leave the green **square to it**, and are dropped if they land within
  `CROSS_MIN_GAP_M` of anything already drawn.

`STREET_MIN_GAP_M` is deliberately small. An 800 m high street already carrying the two
ends of the green has no room for a wide gap, and asking for one does not widen it — it
pushes the side street to the far end of the village, away from the entrance it serves.

Every junction is woven into the line it lands on as a real vertex, on geometry that is
already smoothed: casting at an unsmoothed centreline and emitting the smoothed one
leaves the two a metre apart, close enough to look joined and far enough not to be.

## Coordinates

Local coordinates are playable metres, **x east, y south from the north edge**, origin at
the NW corner. `map_source.local_to_global` / `global_to_local` convert to and from the
lat/lon in the `.osm`, using a flat-earth approximation about the map centre
(52.0620, −1.3400).

The two reference images are 1024 × 1024 and are mapped 1:1 onto the *playable* area, so
one source pixel is 8 m. The DEM's non-playable border is filled by mirroring the source
outwards.

**Resizing the map** is one line: change `PLAYABLE_M` (and `CANVAS_M`) in `map_source.py`.
Everything read off the reference images scales with `map_source.MAP_SCALE`; physical
widths — hedgerows, lane corridors, outline tolerances — deliberately do not.

## OSM tag vocabulary

```
landuse=farmland                              fields
landuse=farmyard                              village, industry pads
natural=wood + landuse=farmyard + leaf_type   woodland
natural=water + waterway=riverbank            the river
highway=primary / secondary / tertiary        road hierarchy
bridge=yes + layer=1                          river crossings
```

Woods carry **both** `natural=wood` and `landuse=farmyard`; anything reading the file
must check `natural=wood` first or it will treat a wood as a yard.

## Running it

Requires `numpy`, `scipy`, `matplotlib` and `pillow`. If the system Python does not have
them:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install numpy scipy matplotlib pillow
```

Then, from the repository root:

```bash
python3 dem_generator/generate_new_dem_12k.py   # heightmap + hillshade previews
python3 osm_generator/generate_osm.py           # map.osm
python3 pf_generator/generate_soil.py           # optional: Precision Farming soil map
python3 visualizer/create_3d_viewer.py          # web assets + dem_viewer_3d.html

cd visualizer && python3 -m http.server 8000    # then open localhost:8000/dem_viewer_3d.html
```

The DEM and OSM generators are independent of each other — both read only
`map_source.py` — so either can be re-run on its own. The visualizer needs both.

### Checks

```bash
python3 dem_generator/measure_elevation.py   # river profile, pad flatness, field slopes
python3 osm_generator/check_forest_nodes.py  # feature inventory from map.osm
python3 osm_generator/visualize_osm.py       # map_osm_visual.png
```

## Tuning

Most of what shapes the map is a named constant:

- `map_source.py` — canvas and playable size, elevation range, river surface levels,
  map centre, and the `LAKE_*` block (where along the river the lake sits, how wide and
  how deep).
- `dem_generator/generate_new_dem_12k.py` — `TREND_BLUR_M` (how much of the reference
  image's field pattern survives as landform), the `VALLEY_*` profile, `PAD_SKIRT_M`.
- `osm_generator/generate_osm.py` — `SPACING_OPEN_M` / `SPACING_DENSE_M` /
  `SPACING_SCALE` and `MAX_FIELDS` (field sizes and count), `HEDGE_M`,
  `LANE_COVERAGE_M`, the river and wood crossing penalties, `SEED`, the
  `WOOD_CLOSE_M` / `WOOD_OPEN_M` / `WOOD_POCKET_M` / `WOOD_SIMPLIFY_M` block that
  decides how blocky the woods come out and how much stray ground they take, and the
  `GATE_MERGE_M` / `STREET_MIN_GAP_M` / `BACK_LANE_OFF_M` / `STREET_INSET_M` /
  `CROSS_SPACING_M` block that shapes the village.

`SIMPLIFY_SLACK_M` is worth understanding before touching the outline tolerances: every
mask a parcel is cut against is grown by it first, because the Douglas-Peucker pass
afterwards can pull an edge back across the boundary it was just cut to.
