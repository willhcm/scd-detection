from scipy.ndimage import distance_transform_edt
import numpy as np

VETO_SCALAR_NAMES = (
    "candidate_mean_slope",
    "annulus_mean_slope",
    "log_context_width_m",
    'depth_width_ratio')

VETO_BAND_INDEX = {'DEM': 0, 'SLOPE': 1, 'MASK': 2} # index map of band names in veto dataset ([DEM Context, Slope Context, Mask from MaskR-CNN])

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

    # scd approx size
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
    """
    Docstring
    """
    
    candidate_mask = np.asarray(candidate_mask).astype(bool)

    valid = np.isfinite(dem)

    if not valid.any():
        return None, None

    # Fill NODATA values in DEM with the median of valid values to avoid NaNs in calculations (handles very small regions where no data exists at edge of DEM scene)
    fill_value = float(np.nanmedian(dem[valid]))
    dem = np.where(valid, dem, fill_value).astype(np.float32)

    candidate_mask, annulus = _make_annulus(candidate_mask)

    # Ensure that the candidate and annulus are within the valid DEM area
    candidate_valid = candidate_mask & valid # where the candidate is valid in the DEM
    annulus_valid = annulus & valid # where the annulus is valid in the DEM

    if not candidate_valid.any():
        candidate_valid = candidate_mask

    # if no valid pixels in annulus, use all valid pixels outside candidate as annulus
    if not annulus_valid.any():
        annulus_valid = valid & ~candidate_mask

    # get values from masks 
    candidate_values = dem[candidate_valid]
    annulus_values = dem[annulus_valid]

    # calculate stats from these values.
    annulus_mean = float(np.mean(annulus_values))

    # repeat for slope
    h, width = dem.shape
    pixel_size_m = float(context_width_m) / float(width)

    grad_y, grad_x = np.gradient(dem, pixel_size_m, pixel_size_m)

    slope = np.hypot(grad_x, grad_y).astype(np.float32)

    candidate_slope = slope[candidate_valid]
    annulus_slope = slope[annulus_valid]

    depth = np.median(candidate_values) - annulus_mean
    depth_width_ratio = depth / candidate_diameter_m 

    # same order as VETO_SCALAR_NAMES, used to reference indices throughout repo.
    scalar_features = np.asarray([
            float(np.mean(candidate_slope)), # avg slope in SCD
            float(np.mean(annulus_slope)), # avg slope in surrounding ring
            float(np.log1p(context_width_m)),
            float(depth_width_ratio)], # context of tile width
              dtype=np.float32)

    # return dem, slope (stacked) and scalsrs
    dem_context = np.stack([dem, slope, candidate_mask.astype(np.float32)], axis=0).astype(np.float32)

    return dem_context, scalar_features

def normalise_veto_data(dem, slope, scalars, norm_stats):

    dem, slope = _handle_outliers(dem, slope)

    dem_rel = dem - np.median(dem)

    scalars = (scalars - norm_stats['scalars'][VETO_SCALAR_NAMES]['median']) / norm_stats['scalars'][VETO_SCALAR_NAMES]['iqr']
    dem_rel = (dem_rel - norm_stats['DEM']['median']) / norm_stats['DEM']['iqr']
    slope = (slope - norm_stats['SLOPE']['median']) / norm_stats['SLOPE']['iqr']
        
    return dem_rel, slope, scalars

def _handle_outliers(dem, slope):

    out = []
    for band in dem, slope:
        valid = np.isfinite(band)

        if not valid.any():
            raise ValueError(f"Band contains no finite pixels")

        fill_value = np.median(band[valid])
        band = np.where(valid, band, fill_value)
        lower_bound = np.percentile(band, 0.5)
        upper_bound = np.percentile(band, 99.5)

        band=  np.clip(band, lower_bound, upper_bound)
        out.append(band)

    return tuple(out)