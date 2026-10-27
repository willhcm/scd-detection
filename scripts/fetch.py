#!/usr/bin/env python
import argparse
import os

from cop30 import Cop30Loader


def parse_zones(text):
    zones = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            zones.extend(range(int(a), int(b) + 1))
        else:
            zones.append(int(part))
    return zones


def main():
    p = argparse.ArgumentParser(description="Fetch Copernicus 30 m DEM tiles by UTM zone")
    p.add_argument("--zones", required=True, help="e.g. 38-44 or 38,40,42")
    p.add_argument("--lat", nargs=2, type=int, required=True, metavar=("S", "N"))
    p.add_argument("--tiles", required=True, help="tile directory")
    p.add_argument("--buffer", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--warp", metavar="OUT_DIR", help="also write one UTM raster per zone")
    p.add_argument("--res", type=float, default=30)
    args = p.parse_args()

    zones = parse_zones(args.zones)
    loader = Cop30Loader(args.tiles, workers=args.workers)
    lat = tuple(args.lat)

    names = set()
    for z in zones:
        names.update(loader.wanted(f"EPSG:326{z:02d}", args.buffer, lat))
    names = sorted(names)
    print(f"{len(zones)} zones, {len(names)} unique tiles", flush=True)
    loader.download(names)
    print("download done", flush=True)

    if args.warp:
        os.makedirs(args.warp, exist_ok=True)
        for z in zones:
            out = os.path.join(args.warp, f"z{z}_{int(args.res)}m.tif")
            if os.path.exists(out):
                continue
            print(f"warping zone {z}", flush=True)
            loader.warp_zone(f"EPSG:326{z:02d}", out, args.res, args.buffer, lat)


if __name__ == "__main__":
    main()