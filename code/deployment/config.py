from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    score_threshold: float = 0.50
    veto_threshold: float = 0.50
    mask_threshold: float = 0.50
    detect_threshold: float = 0.60
    stride_frac : float = 0.5
    tile_size: int = 512
    NO_DATA: int = -9999
    veto_context_tile_size: int =224,
    veto_batch_size: int = 16,
    context_scale: float = 4.0,
    min_context_width_m: float = 500.0,
    max_context_width_m: float = 4000.0,
    to_veto: bool = True


_cfg = None

def init(**kwargs):
    global _cfg
    _cfg = Config(**kwargs)

def get():
    if _cfg is None:
        raise RuntimeError("config.init() hasn't been called")
    return _cfg