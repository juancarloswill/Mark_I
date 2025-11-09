# agentes/log_utils.py
from __future__ import annotations
import os, time, logging, functools
from contextlib import contextmanager

try:
    import psutil  # opcional
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

def setup_logging_from_config(cfg: dict) -> logging.Logger:
    lg_cfg = (cfg or {}).get("logging", {}) or {}
    enable = bool(lg_cfg.get("enable", True))
    logger = logging.getLogger("marki")
    # evita handlers duplicados
    if logger.handlers:
        return logger

    level = str(lg_cfg.get("level", "INFO")).upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    if enable and lg_cfg.get("show_console", True):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if enable and lg_cfg.get("to_file", True):
        path = lg_cfg.get("file_path", "logs/marki.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.disabled = not enable
    return logger

def log_cfg_snapshot(logger: logging.Logger, cfg: dict, keys=("simbolo","timeframe","cantidad_datos","validacion","agent","bt")):
    if logger.disabled: return
    safe = {}
    for k in keys:
        if k in (cfg or {}):
            safe[k] = cfg[k]
    logger.info(f"[CFG] Snapshot: {safe}")

def timeit(logger: logging.Logger):
    """Decorador: logea tiempo de ejecución de una función en DEBUG."""
    def _decor(fn):
        @functools.wraps(fn)
        def _wrap(*args, **kwargs):
            if logger.disabled:
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = (time.perf_counter() - t0) * 1000.0
                logger.debug(f"[TIMER] {fn.__name__} -> {dt:.1f} ms")
        return _wrap
    return _decor

@contextmanager
def timed_block(logger: logging.Logger, label: str):
    """Context manager: logea tiempo de un bloque de código en DEBUG."""
    if logger.disabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        logger.debug(f"[TIMER] block:{label} -> {dt:.1f} ms")

class Heartbeat:
    """Heartbeat para loops largos, con ETA y (opcional) CPU/RAM."""
    def __init__(self, logger: logging.Logger, total: int | None, every: int = 10, with_memory: bool = False):
        self.logger = logger
        self.total = total
        self.every = max(1, int(every))
        self.with_memory = with_memory and _HAS_PSUTIL
        self.t0 = time.perf_counter()
        self.count = 0
        self.proc = None
        if self.with_memory:
            try:
                import psutil
                self.proc = psutil.Process(os.getpid())
            except Exception:
                self.proc = None
                self.with_memory = False

    def step(self, label: str = ""):
        if self.logger.disabled:
            return
        self.count += 1
        if self.count % self.every != 0:
            return
        elapsed = time.perf_counter() - self.t0
        rate = self.count / max(elapsed, 1e-9)
        eta = None
        if self.total:
            remain = self.total - self.count
            eta = remain / max(rate, 1e-9)
        msg = f"[HB] step={self.count}"
        if label: msg += f" | {label}"
        msg += f" | elapsed={elapsed:.1f}s | rate={rate:.2f} it/s"
        if eta is not None: msg += f" | eta~{eta:.1f}s"
        if self.with_memory and self.proc:
            try:
                rss = self.proc.memory_info().rss / (1024**2)
                cpu = self.proc.cpu_percent(interval=0.0)
                msg += f" | mem={rss:.0f}MB | cpu={cpu:.0f}%"
            except Exception:
                pass
        self.logger.info(msg)
