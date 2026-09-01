# osm_generator

`generate_osm.py` writes `map.osm` for the 8192 x 8192 m playable area.

Map centre: 52.0620, -1.3400 (see `map_source.LAT_CENTER` / `LON_CENTER`).
Local coordinates are playable metres, x east, y south from the north edge.

- `generate_osm.py`      build map.osm from the reference images
- `visualize_osm.py`     render map.osm to map_osm_visual.png
- `check_forest_nodes.py` feature inventory (counts, areas, road network)

`custom_osm.osm` is a JOSM re-save of an earlier version of the map, kept for reference.
