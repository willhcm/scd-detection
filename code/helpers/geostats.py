import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path, PosixPath


# Gemini assistance with sns plotting function (specifically stat=density)
# Gemini assistance with formatting]

def load_metrics(filepath):
    """Loads object metrics from a .npz file."""
    data = np.load(filepath)
    metrics = data.keys()
    return {key: data[key] for key in data.files}, metrics

UNITS = {'areas': 'sq meteres',
         'major_axis_length': 'meters',
         'depth': 'meters',
         'nearest_neighbour_distances': 'meters'}

DISPLAY_NAMES = {'weymouth_only': 'Weymouth Chalk Dolines',
                 'wells_only': 'Mendip Hills Karst Sinkholes',
                 'wellington_only': 'Quantock Hills Depressions (near Variscan WCHF)'}

class StatsStack():

    def __init__(self, region_names: list, base_path: PosixPath):

        self.region_names = region_names
        self.base_path = base_path
        self.all_region_metrics = {}
        # probably not the most efficient way of doing this.
        # will break if there are some regions that dont have all the metrics.
        self.metrics = set()
        
        for region_name in self.region_names:
            file_path = os.path.join(self.base_path, f'{region_name}_object_metrics.npz')
            if os.path.exists(file_path):
                self.all_region_metrics[region_name], metrics = load_metrics(file_path)
                self.metrics.update(metrics)
                print(f"Metrics loaded for {region_name}: {self.all_region_metrics[region_name].keys()}")
            else:
                print(f"Warning: {file_path} not found. Skipping region {region_name}.")

    
    def plot_histogram(self, metric_key, unit=None):
        """Plots a comparison histogram for a given metric across multiple datasets."""
        plt.figure(figsize=(12, 7))
        for region_name, metrics in self.all_region_metrics.items():
            # stat=density so that the varying number of samples in each location doesnt distort plot 
            sns.histplot(metrics[metric_key], bins=50, kde=True, stat='density', label=DISPLAY_NAMES[region_name], alpha=0.6)

        plt.title(f'Comparison of {metric_key.replace("_", " ").title()} Distribution')
        x_label = metric_key.replace("_", " ").title()
        if unit:
            x_label += f' ({unit})'
        plt.xlabel(x_label)
        plt.ylabel('Frequency')
        plt.legend(title='Region')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_metrics(self):
        """Plots a histogram for each metric (each on a different figure object.)"""
        for metric in self.metrics:
            self.plot_histogram(metric)
            
