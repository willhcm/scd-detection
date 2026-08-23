import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path, PosixPath
from scipy.stats import gaussian_kde
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from rasterio.transform import array_bounds
from skimage.measure import label, regionprops
from scipy.spatial import KDTree


# Gemini assistance with sns plotting function (specifically stat=density)
# Gemini assistance with formatting]

# used once to generate stats for SCD sizes, but the code subsequently broke and i have no need to fix it :>

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
                print(f"{file_path} not found. Skipping region {region_name}.")

    
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

# analyser class to report statistics for depression morphologies. can be saved to disk, where statsstack can visualise different regions agaisnt eachother
# probably needs an inherent plot function to just see one region, will add in a later push.
class Analyser():

    def __init__(self, pred, dem, points, out_path):
        self.pred = pred
        self.dem = dem
        self.points = points
        self.resolution = 1
        self.out_path = out_path

        self.labeled_pred = label(self.pred)
        # Get properties for each labeled region using skimage!
        self.regions = regionprops(self.labeled_pred)

        # Initialize lists to store ratios
        self.width_length_ratios = []
        self.depth_length_ratios = []
        self.depths = []
        self.areas = []
        self.major_axis_lengths = []
        
        self.nn()
        self.calculate_properties()

    def get_kde_vals(self):
        coords = np.column_stack([
        self.points.geometry.x.values,
        self.points.geometry.y.values
    ])
        self.coords = coords
        kde = gaussian_kde(coords.T, bw_method=0.2)

        self.x_min, self.x_max = coords[:, 0].min(), coords[:, 0].max()
        self.y_min, self.y_max = coords[:, 1].min(), coords[:, 1].max()

        grid_x, grid_y = np.mgrid[
            self.x_min:self.x_max:200j,
            self.y_min:self.y_max:200j
        ]

        grid_coords = np.vstack([
            grid_x.ravel(),
            grid_y.ravel()
        ])

        vals = kde(grid_coords).reshape(200, 200)

        return vals

    def plot_kde_vals(self):
        vals = self.get_kde_vals()
        fig, ax = plt.subplots(figsize=(8, 6))

        hot_r = matplotlib.colormaps.get_cmap("hot_r")
        colors = hot_r(np.linspace(0, 1, 256))
        colors[:, 3] = np.linspace(0, 1, 256)

        cmap = LinearSegmentedColormap.from_list("hot_r_alpha", colors)

        ax.imshow(vals.T, origin="lower", extent=[self.x_min, self.x_max, self.y_min, self.y_max], cmap=cmap)
        ax.scatter(self.coords[:, 0], self.coords[:, 1], s=2,alpha=0.5, label="Blob centroids")
        ax.set_aspect("equal")
        ax.legend()

    def calculate_properties(self):

        # Iterate through regions and calculate ratios
        for props in self.regions:
            # Ensure the region is not too small to avoid noise
            if props.area > 20:
                self.areas.append(props.area * (self.resolution ** 2))

                if props.major_axis_length > 0:
                    self.width_length_ratios.append(props.minor_axis_length / props.major_axis_length)
                    self.major_axis_lengths.append(props.major_axis_length * self.resolution)

                coords = props.coords 
                max_r, max_c = self.dem.shape
                valid_coords_r = np.clip(coords[:, 0], 0, max_r - 1)
                valid_coords_c = np.clip(coords[:, 1], 0, max_c - 1)

                dem_values = self.dem[valid_coords_r, valid_coords_c]

                if len(dem_values) > 0:
                    # Filter out NODATA values if present in DEM, assuming -9999 as NODATA
                    valid_dem_values = dem_values[dem_values != -9999]

                    if len(valid_dem_values) > 1: # need 2 points for diff
                        # Calculate depth as the difference between max and min elevation within the region
                        depth = np.max(valid_dem_values) - np.min(valid_dem_values)
                        self.depths.append(depth)
                        # Convert major_axis_length from pixels to meters using resolution
                        length_in_meters = props.major_axis_length * self.resolution

                        if length_in_meters > 0:
                            self.depth_length_ratios.append(depth / length_in_meters)

    # AI assistance for KDTree functionality (.query specifically)
    def nn(self):

        # Convert points to a NumPy array for KDTree
        centroids_coords = np.array([(p.x, p.y) for p in self.points])

        if len(centroids_coords) > 1:
            # Build a KDTree for efficient nearest neighbor search
            kdtree = KDTree(centroids_coords)

            # Query the KDTree for the distance to the nearest neighbor for each point
            # k=2 to get the two nearest neighbors (itself and the actual nearest)
            # distances will be the distances to the k neighbors
            distances, _ = kdtree.query(centroids_coords, k=2)

            # (distances[:, 1]) will be the distance to the nearest neighbor
            self.nearest_neighbor_distances = distances[:, 1]

    def save(self):
        save_file_path = f'/content/drive/MyDrive/IRP/figs/{self.out_path}_object_metrics.npz'
        np.savez(
            save_file_path,
            areas=np.array(self.areas),
            major_axis_lengths=np.array(self.major_axis_lengths),
            depths=np.array(self.depths),
            width_length_ratios=np.array(self.width_length_ratios),
            depth_length_ratios=np.array(self.depth_length_ratios),
            nearest_neighbor_distances=np.array(self.nearest_neighbor_distances)
        )

        print(f"All object metrics saved to: {save_file_path}")