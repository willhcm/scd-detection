# dataset wrapper for loading data from a directory of images, and analysing the metadata to see distribution of resolutions and geographies
import numpy as np
import glob
from pathlib import Path
import sys
CODE_DIR = Path("../../../code").resolve()
sys.path.insert(0, str(CODE_DIR))
from modelling.MaskRCNN import MaskRCNNDataset

class DatasetWrapper():

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_resolution_distribution = {}
        self.train_geography_distribution = {}
        self.train_positives_distribution = {}

    def generate_dataset(self, test_region: None):
        # region is folded out region

        self.train_paths = []
        self.test_paths = []
        
        if test_region is None:
            # no region specified, use all data for training
            for tile in self.root_dir.glob("*.npz"):
                self.train_paths.append(tile)
            
            self.analyze_dataset()
            return MaskRCNNDataset(self.train_paths)


        else: 
            # LORO-CV run, leave region out of training set and use for testing.
            for tile in self.root_dir.glob("*.npz"):
                region = self.get_region(tile)
                if region != test_region:
                    self.train_paths.append(tile)
                else:
                    self.test_paths.append(tile)

            self.analyze_dataset()
            return MaskRCNNDataset(self.train_paths), MaskRCNNDataset(self.test_paths)
        

    def analyze_dataset(self):
        # analyze the metadata of the dataset to see distribution of resolutions and geographies

        for path in self.train_paths:
            # extract resolution and geography from metadata
            resolution = self.get_resolution(path)
            geography = self.get_region(path)
            positive = self.get_positive(path)

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
            else:
                self.train_positives_distribution['negative'] += 1

    def get_resolution(self, path):
        # extract resolution from metadata of the image at path
        tile = np.load(path, allow_pickle=True)
        resolution = tile['res']
        return resolution

    def get_region(self, path):
        # extract region from metadata of the image at path
        tile = np.load(path, allow_pickle=True)
        region = tile['region']
        return region

    def get_positive(self, path):
        # extract positive label from metadata of the image at path
        tile = np.load(path, allow_pickle=True)
        positive = tile['positive']
        return positive
