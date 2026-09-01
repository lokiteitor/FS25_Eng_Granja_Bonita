#!/usr/bin/env python3
"""Inventory of what is actually in map.osm, by feature type.

A quick sanity pass over the generated file: counts, areas and the biggest features of
each kind, plus the road network by class. Useful for spotting a threshold change that
quietly halved the woodland or doubled the field count.
"""
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_source as ms


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    osm_path = os.path.join(script_dir, "map.osm")
    if not os.path.exists(osm_path):
        print(f"Error: {osm_path} not found. Run generate_osm.py first.")
        return

    root = ET.parse(osm_path).getroot()
    nodes = {int(n.get('id')): ms.global_to_local(float(n.get('lat')),
                                                  float(n.get('lon')))
             for n in root.findall('node')}

    groups = {'farmland': [], 'wood': [], 'farmyard': [], 'water': []}
    roads = Counter()
    road_km = Counter()
    bridges = 0

    for way in root.findall('way'):
        tags = {t.get('k'): t.get('v') for t in way.findall('tag')}
        coords = [nodes[int(nd.get('ref'))] for nd in way.findall('nd')
                  if int(nd.get('ref')) in nodes]
        if len(coords) < 2:
            continue
        name = tags.get('name', '(unnamed)')

        if 'highway' in tags:
            kind = tags['highway']
            roads[kind] += 1
            road_km[kind] += ms.polyline_length(coords) / 1000.0
            if tags.get('bridge') == 'yes':
                bridges += 1
            continue

        # natural=wood is checked before landuse: woods carry both tags.
        if tags.get('natural') == 'wood':
            key = 'wood'
        elif tags.get('natural') == 'water':
            key = 'water'
        elif tags.get('landuse') in ('farmland', 'farmyard'):
            key = tags['landuse']
        else:
            continue
        groups[key].append((name, ms.ring_area_ha(coords), len(coords)))

    print(f"=== {os.path.basename(osm_path)}: {len(nodes)} nodes, "
          f"{len(root.findall('way'))} ways ===")
    playable_ha = ms.PLAYABLE_M ** 2 / 10000.0
    for key in ('farmland', 'wood', 'farmyard', 'water'):
        items = groups[key]
        if not items:
            print(f"\n{key}: none")
            continue
        total = sum(a for _, a, _ in items)
        areas = sorted(a for _, a, _ in items)
        print(f"\n{key}: {len(items)} ways, {total:.0f} ha "
              f"({total / playable_ha * 100:.1f}% of the playable area)")
        print(f"   size ha: min {areas[0]:.1f}  median {areas[len(areas)//2]:.1f}  "
              f"max {areas[-1]:.1f}")
        for name, area, npts in sorted(items, key=lambda z: -z[1])[:5]:
            print(f"   {area:7.1f} ha  {npts:4d} nodes  {name}")

    print(f"\nroads: {sum(roads.values())} ways, "
          f"{sum(road_km.values()):.1f} km, {bridges} bridge(s)")
    for kind in sorted(roads, key=lambda k: -road_km[k]):
        print(f"   {kind:<12} {roads[kind]:3d} ways  {road_km[kind]:6.2f} km")

    # Nodes shared by more than one way are the junctions; a road network with none
    # would mean every way is topologically isolated.
    usage = Counter()
    for way in root.findall('way'):
        for nd in way.findall('nd'):
            usage[int(nd.get('ref'))] += 1
    shared = sum(1 for v in usage.values() if v > 1)
    print(f"\nshared nodes (junctions): {shared}")


if __name__ == '__main__':
    main()
