from scipy.ndimage import distance_transform_edt
import numpy as np

VETO_SCALAR_NAMES = (
    "candidate_mean_minus_annulus_mean_m",
    "candidate_median_minus_annulus_median_m",
    "candidate_p10_minus_annulus_median_m",
    "context_relief_p95_p05_m",
    "candidate_mean_slope",
    "annulus_mean_slope",
    "diameter_context_ratio",
    "log_context_width_m",
)

# ChatGPT assistance with _make_annulus and dem context functions
def _make_annulus(candidate_mask):
    """
    Build an object-size-relative annulus outside the candidate.

    The distances are measured from the candidate boundary in pixels.
    """
    candidate_mask = candidate_mask.astype(bool)

    if not candidate_mask.any():
        candidate_mask = candidate_mask.copy()
        cy = candidate_mask.shape[0] // 2
        cx = candidate_mask.shape[1] // 2
        candidate_mask[cy, cx] = True

    # scd approx siz
    equivalent_radius_px = max(
        np.sqrt(candidate_mask.sum() / np.pi),
        1.0)

    # distance from background to nearest foreground pixel (distance to SCD)
    distance_from_candidate = distance_transform_edt(
        ~candidate_mask)

    # annulus as a ring around SCD
    inner_distance = max(1.0, 0.5 * equivalent_radius_px)
    outer_distance = max(3.0, 2.5 * equivalent_radius_px)

    annulus = (distance_from_candidate >= inner_distance) & (distance_from_candidate <= outer_distance)

    # a fallback for very small or clipped candidates.
    if annulus.sum() < 16:
        annulus = (distance_from_candidate > 0) & (distance_from_candidate <= max(5.0, 4.0 * equivalent_radius_px))

    return candidate_mask, annulus


def build_dem_context(
    dem,
    candidate_mask,
    context_width_m,
    candidate_diameter_m):
    
    candidate_mask = np.asarray(candidate_mask).astype(bool)

    valid = np.isfinite(dem)

    if not valid.any():
        return None, None

    fill_value = float(np.nanmedian(dem[valid]))
    dem = np.where(valid, dem, fill_value).astype(np.float32)

    candidate_mask, annulus = _make_annulus(candidate_mask)

    candidate_valid = candidate_mask & valid
    annulus_valid = annulus & valid

    if not candidate_valid.any():
        candidate_valid = candidate_mask

    if not annulus_valid.any():
        annulus_valid = valid & ~candidate_mask

    if not annulus_valid.any():
        annulus_valid = valid

    candidate_values = dem[candidate_valid]
    annulus_values = dem[annulus_valid]
    context_values = dem[valid]

    annulus_mean = float(np.mean(annulus_values))
    annulus_median = float(np.median(annulus_values))

    context_p05, context_p95 = np.percentile(
        context_values,
        [5, 95],
    )

    context_relief = max(float(context_p95 - context_p05), 1e-3)

    # Relative DEM preserves whether the candidate lies above or below
    # its immediate surroundings.
    relative_dem = dem - annulus_median

    relative_dem_scaled = np.clip(relative_dem / context_relief, -5.0, 5.0)

    height, width = dem.shape
    pixel_size_m = float(context_width_m) / float(width)

    grad_y, grad_x = np.gradient(dem, pixel_size_m, pixel_size_m)

    slope = np.hypot(grad_x, grad_y).astype(np.float32)

    # Robust image scaling while retaining the absolute slope in scalars.
    slope_p95 = max(float(np.percentile(slope[valid], 95)), 1e-4)
    slope_scaled = np.clip(slope / slope_p95, 0.0, 3.0) / 3.0

    candidate_slope = slope[candidate_valid]
    annulus_slope = slope[annulus_valid]

    scalar_features = np.asarray(
        [float(np.mean(candidate_values) - annulus_mean), # relative height
            float(np.median(candidate_values) - annulus_median), # relative height 2
            float(np.percentile(candidate_values, 10) - annulus_median), # relative height 3
            context_relief, 
            float(np.mean(candidate_slope)), # avg slope in SCD
            float(np.mean(annulus_slope)), # avg slope in surrounding ring
            float(candidate_diameter_m / context_width_m), # size of SCD vs tile size
            float(np.log1p(context_width_m))], # context of tile width
              dtype=np.float32)

    # return dem, slope (stacked) and scalsrs
    dem_context = np.stack([relative_dem_scaled, slope_scaled, candidate_mask.astype(np.float32)], axis=0).astype(np.float32)

    return dem_context, scalar_features