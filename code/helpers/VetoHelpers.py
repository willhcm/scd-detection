from scipy.ndimage import distance_transform_edt
import numpy as np
import pyproj


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


def build_veto_data(
    context,
    context_mask,
    context_width_m,
    candidate_diameter_m,
    candidate,
    candidate_mask):
    """
    Docstring
    """
    
    candidate_mask = np.asarray(candidate_mask).astype(bool)
    context_mask = np.asarray(context_mask).astype(bool)

    candidate = np.asarray(candidate)
    context = np.asarray(context)
    valid = np.isfinite(context)

    # generate wider DEM context branch
    if not valid.any():
        return None, None, None

    # Fill NODATA values in DEM with the median of valid values to avoid NaNs in calculations (handles very small regions where no data exists at edge of DEM scene)
    fill_value = float(np.nanmedian(context[valid]))
    context = np.where(valid, context, fill_value).astype(np.float32)
    context_mask, annulus = _make_annulus(context_mask)

    # fill for candidate
    candidate_data_valid = np.isfinite(candidate)
    if not candidate_data_valid.any():
        return None, None, None
    candidate_fill = float(np.median(candidate[candidate_data_valid]))
    candidate = np.where(candidate_data_valid, candidate, candidate_fill).astype(np.float32)

    # Ensure that the candidate and annulus are within the valid DEM area
    context_candidate_valid = context_mask & valid
    candidate_valid = candidate_mask & candidate_data_valid
    annulus_valid = annulus & valid

    if not context_candidate_valid.any():
        context_candidate_valid = context_mask
    if not candidate_valid.any():
        candidate_valid = np.ones(candidate.shape, dtype=bool)

    # if no valid pixels in annulus, use all valid pixels outside candidate as annulus
    if not annulus_valid.any():
        annulus_valid = valid & ~context_mask

    # get values from masks 
    candidate_values = candidate[candidate_valid]
    annulus_values = context[annulus_valid]

    # calculate stats from these values.
    annulus_mean = float(np.mean(annulus_values))

    # repeat for slope
    h, width = context.shape
    pixel_size_m = float(context_width_m) / float(width)

    grad_y, grad_x = np.gradient(context, pixel_size_m, pixel_size_m)

    slope = np.hypot(grad_x, grad_y).astype(np.float32)

    candidate_slope = slope[context_candidate_valid]
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
    dem_context = np.stack([context, slope, context_mask.astype(np.float32)], axis=0).astype(np.float32)
    dem_candidate = np.stack([candidate, candidate_mask.astype(np.float32)], axis=0).astype(np.float32)

    return dem_context, dem_candidate, scalar_features

def normalise_veto_data(data, norm_stats):

    dem = data['dem']
    slope = data['slope']
    scalars = data['scalar_features']
    candidate_dem = data.get('candidate_dem')

    dem, slope = _handle_outliers((dem, slope))

    dem_rel = dem - np.median(dem)
    dem_rel = (dem_rel / norm_stats['DEM']['iqr'])
    slope = (slope - norm_stats['SLOPE']['median']) / norm_stats['SLOPE']['iqr']

    candidate_dem_rel = None
    if candidate_dem is not None:
        candidate_dem = _handle_outliers((candidate_dem,))[0]
        candidate_dem_rel = candidate_dem - np.median(candidate_dem)
        candidate_dem_rel /= norm_stats['Candidate_DEM']['iqr']

    for i, name in enumerate(VETO_SCALAR_NAMES):
        scalars[i] = (scalars[i] - norm_stats[name]['median']) / norm_stats[name]['iqr']

    if candidate_dem_rel is None:
        return dem_rel, slope, scalars
    return dem_rel, slope, scalars, candidate_dem_rel

def _handle_outliers(bands):

    out = []
    for band in bands:
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

def x_y_to_long_lat(x, y, current_crs, target_crs="EPSG:4326"):
    """
    Convert x, y coordinates from current_crs to target_crs.
    """

    transformer = pyproj.Transformer.from_crs(current_crs, target_crs, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat
