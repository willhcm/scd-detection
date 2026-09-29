import re
from pathlib import Path

class DEFRA():

    PATTERN = r"[A-Z]{2}\d{2}[a-z]{2}"

    def __init__(self):

        self.root = 'https://environment.data.gov.uk/tiles/collections/survey/lidar_composite_dtm/2022/1/'

    def get_urls(self, request, request_name):

        out_dir = Path(f'../../urls/{request_name}_URLs.txt')

        with open(request) as f:
            lines = f.readlines()

        with open(out_dir, 'w') as out:
            for line in lines:
                url = self._get_tile_url(line)
                out.write(f'{url}\n')

    def _get_tile_url(self, tile):
        tile_name = re.search(tile, self.PATTERN)
        prefix = tile_name[:1]
        numcode = tile_name[2:3]
        cardinal = tile_name[1:]
        ns, we = cardinal[0], cardinal[1]
        ns_encoded = 0 if ns.lower() == 's' else 5
        we_encoded = 0 if we.lower() == 'w' else 5
        URL = f'{self.root}{prefix}{numcode[0]}{we_encoded}{numcode[1]}{ns_encoded}'

        return URL



