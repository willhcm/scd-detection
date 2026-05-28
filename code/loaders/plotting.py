import matplotlib.pyplot as plt
import math
import numpy as np

CMAPS = {
    'DEM':        'terrain',
    'FIR':        'coolwarm',
    'CI':         'plasma',
    'NDVI':       'Greens',
    'LABELS':     'Greys',
    'R':          'Reds',
    'G':          'Greens',
    'B':          'Blues',
    'DEM_SLOPE':  'magma',
    'TPI_30':     'RdBu_r',
    'TPI_75':     'RdBu_r',
    'DEM_GROUND': 'terrain',
}

def plot_stack(stack, layer_index, figsize=(8, 8)):
    no_plot     = {'R', 'G', 'B'}
    plot_layers = [(name, i) for name, i in layer_index.items() if name not in no_plot]

    n     = len(plot_layers) + 1 # +1 for RGB
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for plot_i, (name, stack_i) in enumerate(plot_layers):
        ax = axes[plot_i]
        im = ax.imshow(stack[stack_i], cmap=CMAPS.get(name, 'viridis'))
        ax.set_title(name)
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.8)

    # RGB composite
    rgb = np.stack([
        stack[layer_index['R']],
        stack[layer_index['G']],
        stack[layer_index['B']],
    ], axis=-1).astype(np.float32)
    p2, p98 = np.percentile(rgb[..., 0], 2), np.percentile(rgb[..., 0], 98)
    rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
    axes[len(plot_layers)].imshow(rgb_norm)
    axes[len(plot_layers)].set_title('RGB')
    axes[len(plot_layers)].axis('off')

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()