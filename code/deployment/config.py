from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DeployConfig:
    dem_path: str
    resolutions: tuple
    device: str = "cpu"
    nodata: float = -9999
    native_res: float = 30

    # tiling
    tile_size: int = 512
    stride_frac: float = 0.75
    sigma_px: int = 12
    batch_size: int = 16

    # detection
    score_threshold: float = 0.65  # tune
    mask_threshold: float = 0.50  # tune
    detect_threshold: float = 0.6  # tune

    # merging across resolutions
    support_threshold: float = 0.65
    min_support: int = 2
    single_scale_keep: float = 0.90
    merge_band: int = 1024
    object_threshold: float = 0.50
    size_scale: float = 1.0
    size_sigma: float = 0.8

    # veto
    to_veto: bool = True
    veto_threshold: float = 0.6  # tune
    veto_batch_size: int = 16
    veto_context_tile_size: int = 224
    veto_object_fraction: float = 0.5
    context_scale: float = 4.0
    min_context_width_m: float = 500.0
    max_context_width_m: float = 4000.0

    # shapefile export
    export_threshold: float = 0.40
    min_object_m2: float = 400.0  # 20m x 20m
    min_native_pixels: int = 25

_cfg = None

def init(**kwargs):
    global _cfg
    _cfg = DeployConfig(**kwargs)
    return _cfg


def update(**kwargs):
    global _cfg
    _cfg = replace(get(), **kwargs)
    return _cfg


def get():
    if _cfg is None:
        raise RuntimeError("config.init() has not been called")
    return _cfg