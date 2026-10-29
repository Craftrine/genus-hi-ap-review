"""Keep JIT and plotting caches in the writable project, not the frozen core."""
import os
from pathlib import Path

_project = Path(__file__).resolve().parents[1]
os.environ.setdefault('NUMBA_CACHE_DIR', str(_project/'tmp/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', str(_project/'tmp/matplotlib'))
