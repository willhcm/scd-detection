import os
from dataclasses import dataclass, replace


def _get_cpu_count():

    # Slurm allocation takes precedence
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")

    if slurm_cpus:
        return int(slurm_cpus)

    # Local process CPU availability
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1

    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))

    return os.cpu_count() or 1


@dataclass(frozen=True)
class SystemConfig:

    # CPU resources
    cpu_count: int = None

    # Processing configuration
    gdal_threads: int = None

    def __post_init__(self):

        if self.cpu_count is None:
            object.__setattr__(self, "cpu_count", _get_cpu_count())

        if self.gdal_threads is None:
            object.__setattr__(self, "gdal_threads", self.cpu_count - 1 if self.cpu_count > 1 else 1)

_cfg = None


def init(**kwargs):
    global _cfg
    _cfg = SystemConfig(**kwargs)
    return _cfg

def update(**kwargs):
    global _cfg
    _cfg = replace(get(), **kwargs)
    return _cfg

def get():
    if _cfg is None:
        raise RuntimeError("system.init() has not been called")
    return _cfg