import re
from pathlib import Path
import geopandas as gpd
import argparse

PATTERN = r"[A-Z]{2}\d{2}[a-z]{2}"
ROOT = 'https://environment.data.gov.uk/tiles/collections/survey/lidar_composite_dtm/2022/1/'

def get_urls_from_list(request, request_name):

    out_dir = Path.cwd() / "urls"
    out_dir.mkdir(exist_ok=True)

    with open(request) as f:
        lines = f.readlines()

    out_path = out_dir / f'{request_name}_urls.txt'
    with open(out_path, 'w') as out:
        for line in lines:
            url = _get_tile_url(line)
            out.write(f'{url}\n')

    print(f'list of urls saved to {out_path}')

def _get_tile_url(tile):
    match = re.search(PATTERN, tile)
    tile_name = match[0]
    
    prefix = tile_name[:2]
    numcode = tile_name[2:4]
    cardinal = tile_name[-2:]
    ns, we = cardinal[0], cardinal[1]
    ns_encoded = 0 if ns.lower() == 's' else 5
    we_encoded = 0 if we.lower() == 'w' else 5
    URL = f'{ROOT}{prefix}{numcode[0]}{we_encoded}{numcode[1]}{ns_encoded}'

    return URL

def get_urls_from_bounds(bounds, request_name):

    out_dir = Path.cwd() / "urls"
    out_dir.mkdir(exist_ok=True)

    index = gpd.read_file('../../../datastore/index.gpkg')
    requested_bounds = gpd.read_file(bounds)

    index = index.to_crs(requested_bounds.crs)

    matches = gpd.sjoin(requested_bounds, index, predicate='intersects')

    out_path = out_dir / f'{request_name}_urls.txt'
    with open(out_path, 'w') as out:
        for tile in matches:
            name = tile['TILE_NAME']
            url = _get_tile_url(name)
            out.write(f'{url}\n')

    print(f'list of urls saved to {out_path}')

def main(request, request_name):
    if Path(request).suffix in ['.txt', '.csv']:
        get_urls_from_list(request, request_name)
    elif Path(request).suffix in ['.geojson', '.gpkg']:
        get_urls_from_bounds(request, request_name)
    else:
        raise ValueError(f"Unsupported file type: {Path(request).suffix}. Please provide a .txt, .csv, .geojson, or .gpkg file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--request", required=True)
    parser.add_argument("--request_name", required=True)

    args = parser.parse_args()

    main(args.request, args.request_name)