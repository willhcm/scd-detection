import numpy as np


# changed to multidirectional hillshade
# multi-directional hillshade. returns a function for registry
def calculate_hillshade(dem, cell_size=1.0, altitude_deg=45.0, z_factor=1.0):

    alt = np.radians(altitude_deg)
    dz_dx = np.gradient(dem * z_factor, cell_size, axis=1)
    dz_dy = np.gradient(dem * z_factor, cell_size, axis=0)
    slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    aspect = np.arctan2(-dz_dy, dz_dx)
    hs = np.zeros_like(dem, dtype=np.float64)
    for az_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        az = np.radians(360 - az_deg + 90)
        hs += np.cos(alt) * np.cos(slope) + np.sin(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs / 8, 0, 1).astype(np.float32)
