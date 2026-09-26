# dataset wrapper for loading data from a directory of images, and analysing the metadata to see distribution of resolutions and geographies
import numpy as np
import glob
from pathlib import Path
import sys
CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))
from modelling.MaskRCNN import MaskRCNNDataset
import matplotlib.pyplot as plt


class DatasetWrapper():

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.train_resolution_distribution = {}
        self.train_geography_distribution = {}
        self.train_positives_distribution = {'positive': 0, 'negative': 0}
        self.geography_positives_distribution = {}
        self.resolutions_positive_distribution = {}

    def generate_dataset(self, test_region: None):
        # region is folded out region

        self.train_paths = []
        self.test_paths = []
        
        if test_region is None:
            # no region specified, use all data for training
            for tile in self.root_dir.glob("*.npz"):
                self.train_paths.append(tile)
            
            self._analyze_dataset()
            return MaskRCNNDataset(self.train_paths)

        else: 
            # LORO-CV run, leave region out of training set and use for testing.
            for tile in self.root_dir.glob("*.npz"):
                region = self._get_region(tile)
                if region != test_region:
                    self.train_paths.append(tile)
                else:
                    self.test_paths.append(tile)

            self._analyze_dataset()
            return MaskRCNNDataset(self.train_paths), MaskRCNNDataset(self.test_paths)

    def plot_distribution(self):

        # plot the distribution of resolutions and geographies in the training set
        # color parts of each hist bar for positive and negative samples in the geography distribution
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        bins = np.arange(0, 30, 2.5)
        axs[0].hist(list(self.train_resolution_distribution.keys()), bins=bins, weights=list(self.train_resolution_distribution.values()))
        axs[0].set_title("Resolution Distribution")
        axs[0].set_xlabel("Resolution (m)")
        axs[0].set_ylabel("Count")
        # color parts of each hist bar for positive and negative samples in the geography distribution
        # with each resolution binned
        for bin in bins:
            midpoint = (bin + bin + 2.5) / 2
            positive_count = 0
            negative_count = 0
            for resolution, count in self.train_resolution_distribution.items():
                if bin <= resolution < bin + 2.5:
                    positive_count += self.resolutions_positive_distribution.get(resolution, {}).get('positive', 0)
                    negative_count += self.resolutions_positive_distribution.get(resolution, {}).get('negative', 0)
            axs[0].bar(midpoint, positive_count, color='green', width=2.5, align='center')
            axs[0].bar(midpoint, negative_count, bottom=positive_count, color='red', width=2.5, align='center')
            axs[0].text(midpoint, positive_count + negative_count / 2, f'{(positive_count+0.0001) / (positive_count+negative_count + 0.0001):.2f}', ha='center', va='center')
        # colour bars in two section based on positive/negative distribution
        axs[1].bar(self.train_geography_distribution.keys(), self.train_geography_distribution.values())
        for i, (geography, count) in enumerate(self.train_geography_distribution.items()):
            positive_count = self.geography_positives_distribution.get(geography, {}).get('positive', 0)
            negative_count = self.geography_positives_distribution.get(geography, {}).get('negative', 0)
            axs[1].bar(geography, positive_count, color='green')
            axs[1].bar(geography, negative_count, bottom=positive_count, color='red')
        axs[1].set_title("Geography Distribution")
        axs[1].set_xlabel("Geography")
        axs[1].set_ylabel("Count")

        axs[2].bar(self.train_positives_distribution.keys(), self.train_positives_distribution.values())
        axs[2].set_title("Positives Distribution")
        axs[2].set_xlabel("Positive/Negative")
        axs[2].set_ylabel("Count") 

        plt.show()

    def _analyze_dataset(self):
        # analyze the metadata of the dataset to see distribution of resolutions and geographies

        for path in self.train_paths:
            # extract resolution and geography from metadata
            resolution = self._get_resolution(path)
            geography = self._get_region(path)
            positive = self._get_positive(path)

            # update resolution distribution
            if resolution not in self.train_resolution_distribution:
                self.train_resolution_distribution[resolution] = 0
            self.train_resolution_distribution[resolution] += 1

            # update geography distribution
            if geography not in self.train_geography_distribution:
                self.train_geography_distribution[geography] = 0
            self.train_geography_distribution[geography] += 1

            # update positives distribution
            if positive == 1:
                self.train_positives_distribution['positive'] += 1
                if geography not in self.geography_positives_distribution:
                    self.geography_positives_distribution[geography] = {'positive': 0, 'negative': 0}
                self.geography_positives_distribution[geography]['positive'] += 1
                if resolution not in self.resolutions_positive_distribution:
                    self.resolutions_positive_distribution[resolution] = {'positive': 0, 'negative': 0}
                self.resolutions_positive_distribution[resolution]['positive'] += 1

            else:
                if geography not in self.geography_positives_distribution:
                    self.geography_positives_distribution[geography] = {'positive': 0, 'negative': 0}
                self.train_positives_distribution['negative'] += 1
                self.geography_positives_distribution[geography]['negative'] += 1
                if resolution not in self.resolutions_positive_distribution:
                    self.resolutions_positive_distribution[resolution] = {'positive': 0, 'negative': 0}
                self.resolutions_positive_distribution[resolution]['negative'] += 1

    def _get_resolution(self, path):
        # extract resolution from metadata of the image at path
        with np.load(path) as tile:
            return tile['res'].item()

    def _get_region(self, path):
        with np.load(path, allow_pickle=True) as tile:
            return tile["region"].item()

    def _get_positive(self, path):
        # extract positive label from metadata of the image at path
        with np.load(path, allow_pickle=True) as tile:
            return tile['positive'].item()

class DatasetRefiner():

    def __init__(self, dir, region):
        self.dir = Path(dir)
        self.region = region
        self.tiles = self._get_region_tiles()

    def refine(self):
        # refine dataset by manually accepting or rejecting tiles based on visual inspection of the image and labels
        for tile in self.tiles:
            tile, path = tile
            self._view_tile(tile)

            answer = input(
                "r=RETAIN, x=REJECT, s=SKIP, q=QUIT: "
            ).strip().lower()

            if answer == "q":
                break

            if answer == "s":
                continue

            if answer == "r":
                continue

            elif answer == "x":
                # delete the tile from the directory
                if path.exists():
                    path.unlink()

            else:
                print("Invalid input; candidate skipped.")
                continue

    def _view_tile(self, tile):
        # view a specific tile
        image = tile['image']
        labels = tile['labels']
        fig, axs = plt.subplots(1, 6, figsize=(15, 10))
        for i, band in enumerate(image):
            axs[i].imshow(band, cmap='gray')
            axs[i].set_title(f"Band {i}")
        axs[5].imshow(labels, cmap='gray')
        axs[5].set_title("Labels")
        plt.show()

    def _get_region_tiles(self):
        # return a list of tiles in the directory that match the region
        tiles = []
        for path in self.dir.glob("*.npz"):
            f = np.load(path, allow_pickle=True)
            if f["region"].item() == self.region:
                tiles.append([f, path])
        return tiles

    