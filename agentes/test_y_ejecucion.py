# agentes/test_y_ejecucion.py
# -*- coding: utf-8 -*-
"""
Agente de análisis, recomendación y validación (hold-out) Top-K.

Uso:
  python -m agentes.test_y_ejecucion \
    --config utils/config_optimizado.yaml \
    --backtest_xlsx outputs/backtest_plots/EURUSD_backtest_consolidado.xlsx \
    --top_k 3 \
    --outdir outputs/validacion \
    [--run_audit]
"""

from __future__ import annotations

import os
import re
import sys
import yaml
import math
import glob
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import logging

# -------- logging utils (no intrusivo) --------
try:
    from agentes.log_utils import setup_logging_from_config, log_cfg_snapshot, timeit, timed_block, Heartbeat
except Exception:
    # fallback mínimo si no está disponible el módulo
    def setup_logging_from_config(cfg: dict):
        logger = logging.getLogger("marki")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
            logger.addHandler(ch)
        return logger
    def log_cfg_snapshot(logger, cfg, keys=None): pass
    def timeit(logger): 
        def _decor(fn):
            def _wrap(*a, **k): return fn(*a, **k)
            return _wrap
        return _decor
    def timed_block(logger, label): 
        from contextlib import contextmanager
        @contextmanager
        def _cm(): yield
        return _cm()
    class Heartbeat:
        def __init__(self, logger, total, every=10, with_memory=False): self.logger=logger; self.total=total; self.every=every; self.count=0
        def step(self, label=""): 
            self.count += 1
            if self.count % max(1, int(self.every)) == 0:
                self.logger.info(f"[HB] step={self.count} | {label}")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --------- imports dinámicos del proyecto ----------
def _import_get_model():
    for mod in ("app.utils.registry", "utils.registry", "registry", "modelos.registry"):
        try:
            m = __import__(mod, fromlist=["get_model"])
            return getattr(m, "get_model")
        except Exception:
            continue
    raise ImportError("No se pudo importar get_model desde utils/registry.py")
get_model = _import_get_model()

def _import_exporters():
    for mod in ("reportes.reportes_excel", "app.reportes.reportes_excel", "reportes_excel"):
        try:
            m = __import__(mod, fromlist=[
                "export_backtest_csv_per_model",
                "export_backtest_excel_consolidado"
            ])
            return (
                getattr(m, "export_backtest_csv_per_model"),
                getattr(m, "export_backtest_excel_consolidado"),
            )
        except Exception:
            continue
    raise ImportError("No se encontraron exportadores en reportes/reportes_excel.py")
export_backtest_csv_per_model, export_backtest_excel_consolidado = _import_exporters()

try:
    from app.eda.eda_crispdm import _ensure_dt_index, _find_close, _resample_ohlc  # type: ignore
except Exception:
    def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_index()
    def _find_close(df: pd.DataFrame) -> str:
        return "Close" if "Close" in df.columns else df.columns[-1]
    def _resample_ohlc(df: pd.DataFrame, freq: str = "D", price_col: str = "Close") -> pd.DataFrame:
        return df

_HAS_MT5 = True
try:
    from conexion.easy_Trading import Basic_funcs as _BF
except Exception as _e:
    _HAS_MT5 = False
    _BF = None
    print(f"⚠️ No se pudo importar Basic_funcs (conexion.easy_Trading): {_e}")

# --------- utils ----------
def _load_config(require_path: str) -> dict:
    p = Path(require_path)
    if not p.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _apply_data_window(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if not isinstance(cfg, dict) or not cfg:
        return df
    mode = str(cfg.get("mode", "last_n_bars")).lower()
    if mode == "date_range":
        start = cfg.get("start", None)
        end   = cfg.get("end", None)
        if start is not None: start = pd.to_datetime(start)
        if end   is not None: end   = pd.to_datetime(end)
        if start is None and end is None:
            return df
        if start is not None and end is not None:
            return df.loc[start:end]
        elif start is not None:
            return df.loc[start:]
        else:
            return df.loc[:end]
    else:
        n = int(cfg.get("n_bars", 0))
        if 0 < n < len(df):
            return df.iloc[-n:]
        return df

@timeit(logging.getLogger("marki"))
def _split_train_valid(price: pd.Series, cfg_valid: dict) -> Tuple[pd.Series, Optional[pd.Series]]:
    if not isinstance(cfg_valid, dict) or not cfg_valid:
        return price, None
    modo = str(cfg_valid.get("modo", "last_n")).lower()
    if modo == "date_range":
        start = cfg_valid.get("start", None)
        end   = cfg_valid.get("end", None)
        if start is not None: start = pd.to_datetime(start)
        if end   is not None: end   = pd.to_datetime(end)
        if start is None and end is None:
            return price, None
        mask_valid = pd.Series(True, index=price.index)
        if start is not None: mask_valid &= (price.index >= start)
        if end   is not None: mask_valid &= (price.index <= end)
    else:
        n = int(cfg_valid.get("n", 0))
        if n <= 0 or n >= len(price):
            return price, None
        mask_valid = pd.Series(False, index=price.index)
        mask_valid.iloc[-n:] = True
    price_valid = price[mask_valid]
    price_train = price[~mask_valid]
    if len(price_train) < 50 or len(price_valid) < 10:
        return price, None
    return price_train, price_valid

@timeit(logging.getLogger("marki"))
def _get_series_from_mt5(config: dict) -> Tuple[pd.DataFrame, str]:
    if not _HAS_MT5:
        raise RuntimeError("MT5 no disponible: no es posible reconstruir la serie.")
    simbolo   = config.get("simbolo", "EURUSD")
    timeframe = config.get("timeframe", "D1")
    cantidad  = int(config.get("cantidad_datos", 1500))
    mt5c = config.get("mt5", {})
    bf = _BF(mt5c.get("login"), mt5c.get("password"), mt5c.get("server"), mt5c.get("path"))  # type: ignore
    print("✅ Conexión establecida con MetaTrader 5 (test_y_ejecucion)")
    df = bf.get_data_for_bt(timeframe, simbolo, cantidad)
    ren = {"open":"Open","high":"High","low":"Low","close":"Close","tick_volume":"TickVolume","real_volume":"Volume","time":"Date"}
    for k,v in ren.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})
    if "Date" in df.columns:
        df = df.set_index("Date")
    df = df.sort_index()
    price_col = _find_close(df)
    df = _ensure_dt_index(df)
    freq_cfg = str(config.get("eda", {}).get("frecuencia_resampleo", "D")).upper()
    with timed_block(logging.getLogger("marki"), "resample_ohlc"):
        df = _resample_ohlc(df, freq=freq_cfg, price_col=price_col)
    df = _apply_data_window(df, config.get("data_window", {}) or {})
    return df, price_col

# --------- leer XLSX backtest ----------
@timeit(logging.getLogger("marki"))
def _read_backtest_consolidado(xlsx_path: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    from openpyxl import load_workbook
    wb = load_workbook(filename=xlsx_path, data_only=True, read_only=True)
    sheets = wb.sheetnames
    if "metrics" not in sheets:
        raise ValueError("El archivo no contiene hoja 'metrics'.")
    ws = wb["metrics"]
    header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    headers = [str(h) if h is not None else "" for h in header]
    metrics = pd.DataFrame(list(ws.iter_rows(min_row=2, values_only=True)), columns=headers)
    per_model: Dict[str, pd.DataFrame] = {}
    for name in sheets:
        if name.lower() in {"metrics", "config_info"}:
            continue
        ws2 = wb[name]
        header2 = next(ws2.iter_rows(min_row=1, max_row=1, values_only=True))
        headers2 = [str(h) if h is not None else "" for h in header2]
        dfm = pd.DataFrame(list(ws2.iter_rows(min_row=2, values_only=True)), columns=headers2)
        if "ds" in dfm.columns:
            dfm["ds"] = pd.to_datetime(dfm["ds"], errors="coerce")
            dfm = dfm.set_index("ds")
            dfm.index.name = "ds"
        per_model[name] = dfm
    return metrics, per_model

# --------- selección TOP-K (HR -> RMSE -> DM simple) ----------
@timeit(logging.getLogger("marki"))
def _pick_topk_models(metrics: pd.DataFrame,
                      per_model: Dict[str, pd.DataFrame],
                      k: int,
                      prefer_col_hr: List[str] = ["Direction_Accuracy","Hit Rate","Hit_Rate","HitRate","DA","DA%"],
                      prefer_col_rmse: List[str] = ["RMSE","rmse"]) -> List[str]:
    if metrics is None or metrics.empty:
        raise ValueError("No hay métricas en 'metrics'.")

    cols = {c.lower(): c for c in metrics.columns}

    hr_col = None
    for c in prefer_col_hr:
        if c in metrics.columns: hr_col = c; break
        if c.lower() in cols:    hr_col = cols[c.lower()]; break
    if hr_col is None:
        raise ValueError("No se encontró columna de Hit Rate / Direction_Accuracy.")

    rmse_col = None
    for c in prefer_col_rmse:
        if c in metrics.columns: rmse_col = c; break
        if c.lower() in cols:    rmse_col = cols[c.lower()]; break
    if rmse_col is None:
        raise ValueError("No se encontró columna RMSE.")

    name_col = None
    for c in ("model","Modelo","name","Name","MODEL"):
        if c in metrics.columns:
            name_col = c; break
    if name_col is None:
        cand = [c for c in metrics.columns if metrics[c].dtype==object]
        name_col = cand[0] if cand else metrics.columns[0]

    df_sorted = metrics.sort_values(by=[hr_col, rmse_col], ascending=[False, True]).reset_index(drop=True)
    topk = df_sorted.head(int(k)).copy()

    def _get_errors(sheet: str) -> Optional[pd.Series]:
        if sheet not in per_model:
            return None
        dfm = per_model[sheet].copy()
        y_true = None; y_pred = None
        for yt in ("y_true","true","Y"):
            if yt in dfm.columns:
                y_true = pd.to_numeric(dfm[yt], errors="coerce"); break
        for yp in ("y_pred","yhat","pred","forecast"):
            if yp in dfm.columns:
                y_pred = pd.to_numeric(dfm[yp], errors="coerce"); break
        if y_true is None or y_pred is None:
            return None
        y_true, y_pred = y_true.align(y_pred, join="inner")
        return (y_true - y_pred).dropna()

    def dm_stat(e1: pd.Series, e2: pd.Series, h: int = 1, power: int = 2) -> float:
        e1, e2 = e1.align(e2, join="inner")
        if len(e1) < 20:
            return 0.0
        d = (e1.abs() ** power) - (e2.abs() ** power)
        d = d - d.mean()
        lag = max(h - 1, 0)
        gamma0 = float((d * d).mean())
        s = gamma0
        for l in range(1, lag + 1):
            w = 1.0 - l / (lag + 1.0)
            gamma = float((d[l:] * d[:-l]).mean())
            s += 2.0 * w * gamma
        return float(d.mean() / math.sqrt(s / len(d))) if s > 0 else 0.0

    i = 0
    chosen = []
    while i < len(topk):
        row = topk.iloc[i]
        same = topk[(topk[hr_col]==row[hr_col]) & (topk[rmse_col]==row[rmse_col])]
        if len(same) == 1:
            chosen.append(str(row[name_col])); i += 1; continue
        names = [str(r[name_col]) for _, r in same.iterrows()]
        errors_map = {n: _get_errors(n) for n in names}
        if any(v is None or v.empty for v in errors_map.values()):
            chosen.extend(names); i += len(same); continue
        mse_avg = {n: float((errors_map[n]**2).mean()) for n in names}
        names_sorted = sorted(names, key=lambda n: mse_avg[n])
        anchor = names_sorted[0]
        out_order = [anchor]
        for cand in names_sorted[1:]:
            _ = abs(dm_stat(errors_map[anchor], errors_map[cand], h=1, power=2))
            out_order.append(cand)
        chosen.extend(out_order); i += len(same)
    return chosen[:int(k)]

# --------- señales y metrica auxiliar ----------
def _compute_signal(yhat: float, last_price: float, config: dict, target: str) -> int:
    bt = (config.get("bt") or {})
    pip_size = float(bt.get("pip_size", 0.0001))
    thr_pips = float(bt.get("threshold_pips", 12.0))
    if target == "returns":
        thr_ret = (thr_pips * pip_size) / max(abs(last_price), 1e-9)
        if yhat >  thr_ret:  return 1
        if yhat < -thr_ret:  return -1
        return 0
    else:
        thr_abs = thr_pips * pip_size
        if (yhat - last_price) >  thr_abs: return 1
        if (yhat - last_price) < -thr_abs: return -1
        return 0

def _fixed_window_1d(series: pd.Series, win: int) -> np.ndarray:
    x = series.astype("float32").to_numpy()
    if len(x) >= win:
        tail = x[-win:]
    else:
        pad = np.full((win - len(x),), x[0] if len(x)>0 else 0.0, dtype="float32")
        tail = np.concatenate([pad, x.astype("float32")], axis=0)
    return tail.astype("float32")

# --------- validación rolling 1-paso ----------
@timeit(logging.getLogger("marki"))
def _roll_validate_on_test(model_name: str,
                           model_params: dict,
                           price_train: pd.Series,
                           price_test: pd.Series,
                           config: dict) -> pd.DataFrame:
    logger = logging.getLogger("marki")
    target = str((config.get("bt", {}) or {}).get("target", "returns")).lower()
    freq   = str((config.get("eda", {}) or {}).get("frecuencia_resampleo", "D"))
    cfg_local = {"target": target, "freq": "H" if freq.upper().startswith("H") else "D"}

    key = model_name.strip().lower()
    cfg_local[key] = model_params or {}

    stride = int((config.get("agent", {}) or {}).get("validation", {}).get("refit_stride", 1))
    stride = max(1, stride)

    logger.info(f"[valid] model={model_name} target={target} stride={stride} params={model_params}")

    model = get_model(key, cfg_local)
    cur   = price_train.copy()

    # --- Intento de preentrenado (LSTM), sin romper flujo existente ---
    skip_initial_fit = False
    if key == "lstm" and bool((model_params or {}).get("reuse_pretrained", False)):
        if hasattr(model, "load_pretrained") and model.load_pretrained():
            logger.info("LSTM: modelo+scaler preentrenados cargados. Se salta fit inicial.")
            skip_initial_fit = True
        else:
            logger.info("LSTM: no se encontró preentrenado util; se entrenará desde cero.")

    # FIT inicial (solo si no cargamos preentrenado)
    if not skip_initial_fit:
        with timed_block(logger, f"fit_inicial:{model_name}"):
            model.fit(cur)

    # Heartbeat para loops largos
    profile_cfg = ((config.get("logging") or {}).get("profile") or {})
    hb = Heartbeat(
        logger=logger,
        total=len(price_test.index),
        every=int(profile_cfg.get("every_n_steps", 10) or 10),
        with_memory=bool(profile_cfg.get("with_memory", False))
    )

    # ---- LSTM: warm-up para “congelar” traza (evitar retracing) ----
    lstm_warm_done = False

    preds = []
    step  = 0
    for ts in price_test.index:
        step += 1

        # (A) update/refit antes de predecir
        if step > 1:
            updated = False
            for meth in ("update", "partial_fit", "fit_partial"):
                if hasattr(model, meth):
                    try:
                        with timed_block(logger, f"update:{model_name}"):
                            getattr(model, meth)(cur)
                        updated = True
                        break
                    except Exception:
                        pass
            force_classic_refit = (key in ("arima", "sarima", "prophet", "rw"))
            if force_classic_refit or (not updated and (step - 1) % stride == 0):
                try:
                    with timed_block(logger, f"refit:{model_name}"):
                        model.fit(cur)
                except Exception:
                    pass

        # (B) predecir 1-paso
        with timed_block(logger, f"predict:{model_name}"):
            if key == "lstm":
                win = int((model_params or {}).get("window", 64))
                w1d = _fixed_window_1d(cur, win)

                # warm-up (una sola vez)
                if not lstm_warm_done:
                    try:
                        _ = model.predict(1, last_timestamp=cur.index[-1], last_window=w1d)
                    except TypeError:
                        _ = model.predict(1)
                    lstm_warm_done = True

                try:
                    dfp = model.predict(1, last_timestamp=cur.index[-1], last_window=w1d)
                    if not isinstance(dfp, pd.DataFrame):
                        dfp = pd.DataFrame({"y_pred": [float(np.ravel(dfp)[0])]}, index=pd.DatetimeIndex([ts]))
                except TypeError:
                    dfp = model.predict(1)
            else:
                try:
                    dfp = model.predict(h=1, last_timestamp=cur.index[-1])
                except TypeError:
                    dfp = model.predict(1)

        if "yhat" in getattr(dfp, "columns", []):
            dfp = dfp.rename(columns={"yhat":"y_pred"})
        dfp.index = pd.DatetimeIndex([ts])

        # (C) conversión returns/price coherente
        last_price = float(cur.iloc[-1])
        yhat = float(dfp["y_pred"].iloc[0])
        if target == "returns":
            # modelos base devuelven nivel => a retorno aproximado
            if key in ("arima","sarima","prophet","rw","lstm"):
                yhat = (yhat / max(last_price, 1e-9)) - 1.0
                dfp.loc[dfp.index, "y_pred"] = yhat

        # señal y precio predicho
        signal = _compute_signal(yhat, last_price, config, target)
        dfp["signal"] = int(signal)
        dfp["y_pred_price"] = last_price * (1.0 + yhat) if target == "returns" else float(yhat)

        # ventana usada
        try:
            dfp["fecha_inicio_ventana"] = cur.index.min()
            dfp["fecha_fin_ventana"] = cur.index.max()
        except Exception:
            pass

        preds.append(dfp[["y_pred", "y_pred_price", "signal", "fecha_inicio_ventana", "fecha_fin_ventana"]])

        # (D) revelar obs real y extender ventana
        cur = pd.concat([cur, price_test.loc[[ts]]])

        # heartbeat
        hb.step(label=f"{model_name} rolling")

    out = pd.concat(preds).sort_index()
    out.index.name = "ds"
    return out

# --------- export espejo backtest ----------
@timeit(logging.getLogger("marki"))
def _export_validation(symbol: str,
                       pred_map_valid: Dict[str, pd.DataFrame],
                       price_valid: pd.Series,
                       config: dict,
                       outdir: Path,
                       excel_path: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    export_backtest_csv_per_model(
        symbol=symbol,
        pred_map=pred_map_valid,
        price=price_valid,
        outdir=outdir,
        target=str((config.get("bt",{}) or {}).get("target","returns")).lower(),
        pip_size=float((config.get("bt",{}) or {}).get("pip_size", 0.0001)),
        threshold_mode=str((config.get("bt",{}) or {}).get("threshold_mode","fixed")).lower(),
        threshold_pips=float((config.get("bt",{}) or {}).get("threshold_pips", 12.0)),
        horizon=1,
        initial_train=None,
    )
    export_backtest_excel_consolidado(
        symbol=symbol,
        pred_map=pred_map_valid,
        price=price_valid,
        excel_path=excel_path,
        target=str((config.get("bt",{}) or {}).get("target","returns")).lower(),
        pip_size=float((config.get("bt",{}) or {}).get("pip_size", 0.0001)),
        threshold_mode=str((config.get("bt",{}) or {}).get("threshold_mode","fixed")).lower(),
        threshold_pips=float((config.get("bt",{}) or {}).get("threshold_pips", 12.0)),
        horizon=1,
        annualization=None,
        per_model_params=None,
        config_info={"validacion": config.get("validacion", {})},
        seasonality_m=1,
        initial_train=None,
    )
    logging.getLogger("marki").info(f"💾 Validación (hold-out) exportada en {excel_path}")

def _write_extra_sheets(backtest_xlsx: str, validation_xlsx: Path, metrics_valid_df: pd.DataFrame) -> None:
    logger = logging.getLogger("marki")
    cfg_df = None
    try:
        xls = pd.ExcelFile(backtest_xlsx)
        if 'config_info' in xls.sheet_names:
            cfg_df = pd.read_excel(xls, 'config_info')
    except Exception as e:
        logger.info(f"ℹ️ No fue posible leer 'config_info' del backtest: {e}")

    try:
        with pd.ExcelWriter(validation_xlsx, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            if metrics_valid_df is not None and not metrics_valid_df.empty:
                metrics_valid_df.to_excel(writer, sheet_name='metrics_valid', index=False)
            if cfg_df is not None:
                cfg_df.to_excel(writer, sheet_name='config_info', index=False)
        logger.info(f"💾 Hojas adicionales escritas en {validation_xlsx} (metrics_valid, config_info)")
    except FileNotFoundError:
        with pd.ExcelWriter(validation_xlsx, engine='openpyxl', mode='w') as writer:
            if metrics_valid_df is not None and not metrics_valid_df.empty:
                metrics_valid_df.to_excel(writer, sheet_name='metrics_valid', index=False)
            if cfg_df is not None:
                cfg_df.to_excel(writer, sheet_name='config_info', index=False)
        logger.info(f"💾 Archivo de validación creado y hojas escritas en {validation_xlsx}")

# --------- métricas valid ----------
def _series_pred(df_pred: pd.DataFrame) -> pd.Series:
    tmp = df_pred.copy()
    if not isinstance(tmp.index, pd.DatetimeIndex) and "ds" in tmp.columns:
        tmp = tmp.set_index("ds")
    if "y_pred" in tmp.columns:
        return tmp["y_pred"].astype(float)
    elif "yhat" in tmp.columns:
        return tmp["yhat"].astype(float)
    for c in reversed(list(tmp.columns)):
        s = pd.to_numeric(tmp[c], errors="coerce")
        if s.notna().sum() > 0:
            return s
    raise ValueError("No se encontró columna de predicción en DF.")

def _dm_pvalue_test(y_true: pd.Series, yhat_a: pd.Series, yhat_b: pd.Series) -> float | None:
    y_true, yhat_a = y_true.align(yhat_a, join="inner")
    y_true, yhat_b = y_true.align(yhat_b, join="inner")
    yhat_a = yhat_a.reindex(y_true.index).astype(float)
    yhat_b = yhat_b.reindex(y_true.index).astype(float)
    if len(y_true) < 20:
        return None
    e1 = (y_true - yhat_a).astype(float)
    e2 = (y_true - yhat_b).astype(float)
    d = (e1.abs()**2) - (e2.abs()**2)
    d = d - d.mean()
    gamma0 = float((d * d).mean())
    if gamma0 <= 0: return None
    dm_stat = float(d.mean()) / math.sqrt(gamma0 / len(d))
    from math import erf, sqrt
    def _phi_cdf(z): return 0.5 * (1 + erf(z / sqrt(2)))
    p = 2.0 * (1.0 - _phi_cdf(abs(dm_stat)))
    return p

@timeit(logging.getLogger("marki"))
def _metrics_from_pred_map(pred_map_valid: Dict[str, pd.DataFrame], price_valid: pd.Series) -> pd.DataFrame:
    rows = []
    for name, dfp in (pred_map_valid or {}).items():
        tmp = dfp.copy()
        if not isinstance(tmp.index, pd.DatetimeIndex) and 'ds' in tmp.columns:
            tmp = tmp.set_index('ds')
        yhat = (tmp['y_pred'] if 'y_pred' in tmp.columns else tmp.iloc[:, -1]).astype(float)
        y_true = price_valid.reindex(yhat.index).ffill().dropna()
        yhat = yhat.reindex(y_true.index).astype(float)
        err = (y_true - yhat).astype(float)
        mae = err.abs().mean()
        rmse = math.sqrt((err**2).mean()) if len(err) else float('nan')
        mape = (err.abs() / y_true.replace(0, pd.NA)).astype(float).mean() * 100.0
        dy_true = y_true.diff().fillna(0).apply(lambda x: 1 if x>0 else (-1 if x<0 else 0))
        dy_pred = yhat.diff().fillna(0).apply(lambda x: 1 if x>0 else (-1 if x<0 else 0))
        da = (dy_true == dy_pred).mean() * 100.0
        rows.append({'model': str(name), 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'Direction_Accuracy': da})
    dfm = pd.DataFrame(rows)
    if not dfm.empty:
        dfm = dfm.sort_values(by=['Direction_Accuracy','RMSE'], ascending=[False, True]).reset_index(drop=True)
        dfm.insert(0, 'rank', range(1, len(dfm)+1))
    return dfm

# ============================================================
# =======  BLOQUE DE AUDITORÍA (opcional, no intrusivo)  =====
# ============================================================

_POSSIBLE_SEPS = [",", ";", "\t", "|"]
_KEYVAL_PATTERN = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([^,\s\|;]+)")

def _discover_validacion_csvs(base_dir: str | Path) -> List[str]:
    """Busca CSVs en la carpeta de validación (ARIMA/LSTM/PROPHET/RW primero)."""
    base = Path(base_dir).expanduser().resolve()
    if not base.exists():
        return []
    patrones = [
        str(base / "*ARIMA*.csv"),
        str(base / "*LSTM*.csv"),
        str(base / "*PROPHET*.csv"),
        str(base / "*RW*.csv"),
        str(base / "*.csv"),
    ]
    vistos, hallados = set(), []
    for pat in patrones:
        for p in glob.glob(pat):
            ap = str(Path(p).resolve())
            if ap not in vistos:
                vistos.add(ap)
                hallados.append(ap)
    return hallados

def _normalize_common(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres y tipos más comunes (y_true/y_pred/etc.)."""
    rename_map = {
        "ds":"ds", "date":"ds", "timestamp":"ds", "time":"ds",
        "ytrue":"y_true", "yTrue":"y_true",
        "ypred":"y_pred", "yPred":"y_pred",
        "pred":"y_pred", "price_pred":"y_pred_price",
    }
    for c in list(df.columns):
        cc = c.strip()
        if cc in rename_map:
            df = df.rename(columns={c: rename_map[cc]})
        elif cc.lower() in rename_map:
            df = df.rename(columns={c: rename_map[cc.lower()]})
    # index temporal si existe
    for c in ["ds","date","timestamp","time"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                df = df.set_index(c)
                break
            except Exception:
                pass
    # coerción numérica de columnas de interés
    for c in ["y_true","y_pred","y_pred_price","error","abs_error","sq_error","signal","direction_pred","direction_true"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _try_sep_read(path: str) -> pd.DataFrame | None:
    for sep in _POSSIBLE_SEPS:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return None

def _try_json_per_line(path: str) -> pd.DataFrame | None:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                try:
                    obj = json.loads(line.replace("'", '"'))
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
    if rows:
        return pd.DataFrame(rows)
    return None

def _try_keyval_parse(path: str) -> pd.DataFrame | None:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            kv = dict(_KEYVAL_PATTERN.findall(line))
            if kv:
                rows.append(kv)
    if rows:
        return pd.DataFrame(rows)
    return None

def load_backtest_csv(path: str) -> pd.DataFrame | None:
    """Carga robusta de CSVs (separadores, JSON por línea o pares clave=valor)."""
    if not os.path.isfile(path):
        return None
    # 1) separadores
    df = _try_sep_read(path)
    if df is not None and df.shape[1] > 1:
        return _normalize_common(df)
    # 2) json per line
    df = _try_json_per_line(path)
    if df is not None:
        return _normalize_common(df)
    # 3) key=value
    df = _try_keyval_parse(path)
    if df is not None:
        return _normalize_common(df)
    # 4) fallback extremo: partir una columna por separadores
    try:
        df_raw = pd.read_csv(path, header=None, names=["raw"])
        for sep in _POSSIBLE_SEPS:
            parts = df_raw["raw"].str.split(sep, expand=True)
            if parts.shape[1] >= 2:
                df_try = parts.copy()
                header_row = df_try.iloc[0].astype(str).str.strip().tolist()
                if all(h and h.replace(" ","").isalpha() for h in header_row):
                    df_try.columns = header_row
                    df_try = df_try.iloc[1:].reset_index(drop=True)
                return _normalize_common(df_try)
    except Exception:
        pass
    return None

def infer_scale(series: pd.Series) -> str:
    s = pd.Series(series).dropna().astype(float)
    if len(s) == 0:
        return "unknown"
    med = float(s.abs().median())
    maxv = float(s.abs().max())
    if med < 0.01 and maxv < 0.2:
        return "returns"
    if 0.5 <= float(s.mean()) <= 5.0:
        return "price"
    return "unknown"

def basic_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    yt = pd.Series(y_true).astype(float).to_numpy()
    yp = pd.Series(y_pred).astype(float).to_numpy()
    m = ~np.isnan(yt) & ~np.isnan(yp)
    yt = yt[m]; yp = yp[m]
    if len(yt) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(np.abs(err) / np.where(np.abs(yt) < 1e-12, np.nan, np.abs(yt))) * 100.0)
    return {"n": int(len(yt)), "MAE": mae, "RMSE": rmse, "MAPE": mape}

def describe_signals(sig: pd.Series) -> Dict[str, float]:
    s = pd.Series(sig).dropna().astype(float)
    n = len(s)
    if n == 0:
        return {"n": 0, "p_buy": np.nan, "p_hold": np.nan, "p_sell": np.nan}
    return {
        "n": n,
        "p_buy": float((s == 1).mean() * 100.0),
        "p_hold": float((s == 0).mean() * 100.0),
        "p_sell": float((s == -1).mean() * 100.0)
    }

def _run_audit_on_outdir(outdir: Path, pip_size: float = 0.0001, out_xlsx: Optional[Path] = None) -> None:
    """
    Audita los CSV exportados en `outdir` y escribe:
    - backtest_ranges: métricas por archivo
    - signals_distribution: distribución de señales
    - dm_tests: comparación simple entre modelos (si procede)
    """
    logger = logging.getLogger("marki")
    csvs = _discover_validacion_csvs(outdir)
    if not csvs:
        logger.info("ℹ️ Auditoría: no se encontraron CSVs en la carpeta de validación.")
        return

    bt_rows, sig_rows = [], []
    model_errors: Dict[str, pd.Series] = {}

    for path in csvs:
        name = os.path.basename(path)
        df = load_backtest_csv(path)
        if df is None or df.empty:
            continue

        cols = set(df.columns)
        row = {"file": name, "n_rows": int(len(df))}
        ycols = [c for c in ["y_true", "y_pred", "y_pred_price"] if c in cols]
        scols = [c for c in ["signal", "direction_pred", "direction_true"] if c in cols]
        row["y_cols"] = ",".join(ycols)
        row["signal_cols"] = ",".join(scols)

        if "y_true" in df.columns and "y_pred" in df.columns:
            mets = basic_metrics(df["y_true"], df["y_pred"])
            row.update({f"pred_{k}": v for k, v in mets.items()})
            row["scale_true"] = infer_scale(df["y_true"])
            row["scale_pred"] = infer_scale(df["y_pred"])
            # para DM
            yt, yp = df["y_true"].align(df["y_pred"], join="inner")
            e = (yt - yp).dropna()
            if len(e) > 10:
                model_errors[os.path.splitext(name)[0]] = e

        if "y_true" in df.columns and "y_pred_price" in df.columns:
            mets_p = basic_metrics(df["y_true"], df["y_pred_price"])
            row.update({f"pred_price_{k}": v for k, v in mets_p.items()})

        bt_rows.append(row)

        for sc in scols:
            d = describe_signals(df[sc])
            sig_rows.append({"file": name, "signal_col": sc, **d})

    backtests_df = pd.DataFrame(bt_rows)
    if not backtests_df.empty and "file" in backtests_df.columns:
        backtests_df = backtests_df.sort_values(["file"]).reset_index(drop=True)

    signals_df = pd.DataFrame(sig_rows)
    if not signals_df.empty and {"file","signal_col"}.issubset(signals_df.columns):
        signals_df = signals_df.sort_values(["file", "signal_col"]).reset_index(drop=True)

    # DM simple entre errores (normal aprox.)
    dm_out = []
    keys = list(model_errors.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            e1, e2 = model_errors[a].align(model_errors[b], join="inner")
            if len(e1) < 20:
                continue
            d = (e1.abs()**2) - (e2.abs()**2)
            d = d - d.mean()
            gamma0 = float((d * d).mean())
            if gamma0 <= 0:
                continue
            dm_stat = float(d.mean()) / math.sqrt(gamma0 / len(d))
            # p-valor normal aprox
            from math import erf, sqrt
            def _phi_cdf(z): return 0.5 * (1 + erf(z / sqrt(2)))
            p = 2.0 * (1.0 - _phi_cdf(abs(dm_stat)))
            dm_out.append({"A": a, "B": b, "DM_stat": dm_stat, "p_value": p})

    dm_df = pd.DataFrame(dm_out)

    # escribir excel de auditoría
    if out_xlsx is None:
        out_xlsx = outdir / "auditoria_validacion.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as wr:
        if not backtests_df.empty:
            backtests_df.to_excel(wr, index=False, sheet_name="backtest_ranges")
        if not signals_df.empty:
            signals_df.to_excel(wr, index=False, sheet_name="signals_distribution")
        if not dm_df.empty:
            dm_df.to_excel(wr, index=False, sheet_name="dm_tests")
    logger.info(f"✅ Auditoría escrita en {out_xlsx}")

# =========================
# --------- main ----------
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta a config.yaml (obligatoria)")
    ap.add_argument("--backtest_xlsx", required=True, help="Ruta al XLSX consolidado del backtest")
    ap.add_argument("--outdir", default="outputs/validacion", help="Carpeta de salida para validación")
    ap.add_argument("--top_k", type=int, default=1, help="Numero de mejores modelos a validar (TOP-K)")
    ap.add_argument("--run_audit", action="store_true", help="Si se indica, corre auditoría sobre los CSV exportados")
    args = ap.parse_args()

    config = _load_config(args.config)

    # === Logger desde YAML ===
    logger = setup_logging_from_config(config)
    log_cfg_snapshot(logger, config)

    logger.info("Leyendo backtest consolidado…")
    metrics, per_model = _read_backtest_consolidado(args.backtest_xlsx)

    logger.info("Seleccionando Top-K modelos…")
    topk_names = _pick_topk_models(metrics, per_model, k=max(1, args.top_k))
    logger.info(f"⭐ TOP-{len(topk_names)} modelos (HR→RMSE→DM): {topk_names}")

    logger.info("Descargando serie MT5 y preparando split…")
    df, price_col = _get_series_from_mt5(config)
    price = df[price_col].astype(float)
    price_train, price_test = _split_train_valid(price, config.get("validacion", {}) or {})
    if price_test is None:
        logger.warning("⚠️ No hay bloque de test definido en config['validacion']. Se detiene la validación.")
        sys.exit(0)
    logger.info(f"Split -> train={len(price_train)}, test={len(price_test)} | fechas: train[{price_train.index.min()}..{price_train.index.max()}] test[{price_test.index.min()}..{price_test.index.max()}]")

    name2params_cfg = {str(m.get("name","")).strip().lower(): (m.get("params", {}) or {}) for m in (config.get("modelos") or [])}

    pred_map_valid: Dict[str, pd.DataFrame] = {}
    for name_raw in topk_names:
        key = name_raw.strip().lower()
        params_cfg = name2params_cfg.get(key, {})
        logger.info(f"⚙️ {name_raw}: usando parámetros de config -> {params_cfg}")
        pred_valid = _roll_validate_on_test(name_raw, params_cfg, price_train, price_test, config)
        pred_map_valid[name_raw] = pred_valid
        logger.info(f"[VALID] Terminado {name_raw}: {len(pred_valid)} filas")

    symbol = re.sub(r"[^A-Za-z0-9_]+","_", str(config.get("simbolo","SYMB")).upper())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    excel_path = outdir / "validacion_consolidado.xlsx"

    _export_validation(symbol, pred_map_valid, price_test, config, outdir, excel_path)

    metrics_valid = _metrics_from_pred_map(pred_map_valid, price_test)
    if len(pred_map_valid) >= 1 and not metrics_valid.empty and "model" in metrics_valid.columns:
        top1 = list(pred_map_valid.keys())[0]
        yhat_top1 = _series_pred(pred_map_valid[top1])
        dm_pvals = {}
        for name, dfp in pred_map_valid.items():
            if name == top1:
                dm_pvals[name] = 0.0
            else:
                p = _dm_pvalue_test(price_test, _series_pred(dfp), yhat_top1)
                dm_pvals[name] = float(p) if p is not None else np.nan
        metrics_valid["DM_pvalue"] = metrics_valid["model"].map(dm_pvals)

    _write_extra_sheets(args.backtest_xlsx, excel_path, metrics_valid)
    logger.info("✅ Proceso de validación finalizado.")

    # -------- Auditoría opcional sobre los CSV recién exportados --------
    if args.run_audit:
        bt_cfg = (config.get("bt") or {})
        pip_size = float(bt_cfg.get("pip_size", 0.0001))
        _run_audit_on_outdir(outdir=outdir, pip_size=pip_size, out_xlsx=outdir / "auditoria_validacion.xlsx")

    # --- RESUMEN EJECUTIVO (no intrusivo) ---
    try:
        valid_xlsx_path = os.path.join(args.outdir, "validacion_consolidado.xlsx")
        write_summary_text(valid_xlsx_path, args.outdir)
    except Exception as e:
        print(f"[WARN] Resumen ejecutivo no generado: {e}")

# =================== RESUMEN EJECUTIVO (no intrusivo) ===================

import os, glob, re, json
import numpy as np
import pandas as pd
from pathlib import Path

_POSSIBLE_SEPS = [",", ";", "\t", "|"]
_KEYVAL_PATTERN = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([^,\s\|;]+)")

def _try_sep_read(path: str) -> pd.DataFrame | None:
    for sep in _POSSIBLE_SEPS:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return None

def _try_json_per_line(path: str) -> pd.DataFrame | None:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                try:
                    obj = json.loads(line.replace("'", '"'))
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
    if rows:
        return pd.DataFrame(rows)
    return None

def _try_keyval_parse(path: str) -> pd.DataFrame | None:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            kv = dict(_KEYVAL_PATTERN.findall(line))
            if kv:
                rows.append(kv)
    if rows:
        return pd.DataFrame(rows)
    return None

def _normalize_common(df: pd.DataFrame) -> pd.DataFrame:
    # normaliza nombres comunes y tipos
    rename_map = {
        "ds":"ds", "date":"ds", "timestamp":"ds", "time":"ds",
        "ytrue":"y_true", "yTrue":"y_true",
        "ypred":"y_pred", "yPred":"y_pred",
        "pred":"y_pred", "price_pred":"y_pred_price",
        "direction_pred":"direction_pred",
        "direction_true":"direction_true",
        "signal":"signal"
    }
    for c in list(df.columns):
        cc = str(c).strip()
        if cc in rename_map:
            df = df.rename(columns={c: rename_map[cc]})
        elif cc.lower() in rename_map:
            df = df.rename(columns={c: rename_map[cc.lower()]})
    # index temporal si existe
    for c in ["ds","date","timestamp","time"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                df = df.set_index(c)
                break
            except Exception:
                pass
    # coerción numérica
    for c in ["y_true","y_pred","y_pred_price","error","abs_error","sq_error","signal","direction_pred","direction_true"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _load_validacion_csvs(validacion_dir: str | Path) -> dict[str, pd.DataFrame]:
    base = Path(validacion_dir).expanduser().resolve()
    out: dict[str, pd.DataFrame] = {}
    patrones = ["*ARIMA*backtest*.csv", "*LSTM*backtest*.csv", "*PROPHET*backtest*.csv", "*RW*backtest*.csv"]
    files = []
    for pat in patrones:
        files.extend(glob.glob(str(base / pat)))
    # fallback: cualquier csv
    if not files:
        files = glob.glob(str(base / "*.csv"))
    for p in files:
        name = Path(p).stem  # p.ej. EURUSD_ARIMA_backtest
        # carga robusta
        df = _try_sep_read(p)
        if df is None:
            df = _try_json_per_line(p)
        if df is None:
            df = _try_keyval_parse(p)
        if df is None:
            try:
                raw = pd.read_csv(p, header=None, names=["raw"])
                # explotar por separador si se detecta
                best = None
                for sep in _POSSIBLE_SEPS:
                    parts = raw["raw"].str.split(sep, expand=True)
                    if parts.shape[1] >= 2:
                        best = parts
                        break
                df = best if best is not None else raw
            except Exception:
                continue
        df = _normalize_common(df)
        out[name] = df
    return out

def _direction_accuracy(df: pd.DataFrame) -> float | None:
    """
    Calcula acierto direccional:
    - Si existen 'direction_pred' y 'direction_true' en {-1,0,1}, usa coincidencia exacta.
    - Si no, usa signo de diferencias: sign(y_pred - y_{t-1}) vs sign(y_true - y_{t-1}) si es posible,
      o sign(y_pred - y_true) como último recurso (menos ideal).
    Devuelve porcentaje (0-100).
    """
    try:
        s = df.dropna()
        if "direction_pred" in s.columns and "direction_true" in s.columns:
            d = s[["direction_pred","direction_true"]].dropna()
            if len(d) == 0:
                return None
            return float((d["direction_pred"] == d["direction_true"]).mean() * 100.0)
        # fallback: signos por primera diferencia
        if "y_true" in s.columns and "y_pred" in s.columns:
            y_true = s["y_true"].astype(float)
            y_pred = s["y_pred"].astype(float)
            # intentamos comparar contra y_{t-1} (requiere índice ordenado)
            if y_true.index.is_monotonic_increasing and len(y_true) > 1:
                yt_diff = y_true.diff()
                yp_diff = y_pred.diff()
                m = (~yt_diff.isna()) & (~yp_diff.isna())
                if m.sum() > 0:
                    return float((np.sign(yt_diff[m]) == np.sign(yp_diff[m])).mean() * 100.0)
            # última opción: coincidencia de signo del error (no ideal, pero informativa)
            err = y_pred - y_true
            return float((np.sign(err) == 0).mean() * 100.0)  # casi siempre ~0
    except Exception:
        return None
    return None

def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    try:
        yt = pd.to_numeric(y_true, errors="coerce").to_numpy()
        yp = pd.to_numeric(y_pred, errors="coerce").to_numpy()
        m = ~np.isnan(yt) & ~np.isnan(yp)
        if m.sum() == 0:
            return None
        return float(np.sqrt(np.mean((yt[m] - yp[m])**2)))
    except Exception:
        return None

def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = a.dropna().index.intersection(b.dropna().index)
    return a.loc[idx], b.loc[idx]

from scipy.stats import t as student_t

def _dm_test(ea: pd.Series, eb: pd.Series, h: int = 1, power: int = 2) -> tuple[float | None, float | None]:
    # DM clásico con corrección HAC simple
    try:
        ea, eb = _align(ea, eb)
        if len(ea) < 5 or len(eb) < 5:
            return None, None
        if power == 1:
            la = np.abs(ea)
            lb = np.abs(eb)
        else:
            la = ea**2
            lb = eb**2
        d = la - lb
        T = len(d)
        dbar = float(np.mean(d))

        def acov(x, k):
            x = x - np.mean(x)
            return float(np.sum(x[:T-k] * x[k:]) / T)

        var_d = acov(d, 0)
        for k in range(1, h):
            gam = acov(d, k)
            var_d += 2 * (1 - k/h) * gam
        if var_d <= 0:
            return None, None
        DM = dbar / np.sqrt(var_d / T)
        p = 2.0 * (1.0 - student_t.cdf(abs(DM), df=max(T-1, 1)))
        return float(DM), float(p)
    except Exception:
        return None, None

def write_summary_text(validacion_xlsx_path: str, validacion_dir: str) -> None:
    """
    Genera/actualiza hoja 'summary_text' en el Excel de validación con:
    - Ranking por Direction_Accuracy y RMSE
    - Recomendación automática
    - Resultado de DM (mejor vs segundo)
    """
    xlsx = Path(validacion_xlsx_path).resolve()
    base = Path(validacion_dir).resolve()
    if not xlsx.exists():
        print(f"[WARN] No existe Excel de validación: {xlsx}")
        return

    # 1) Cargar CSVs procesados
    by_model = _load_validacion_csvs(base)
    if not by_model:
        print(f"[WARN] No se encontraron CSVs en {base}")
        return

    rows = []
    err_series: dict[str, pd.Series] = {}  # para DM
    for name, df in by_model.items():
        # inferir nombre corto del modelo (ARIMA/LSTM/PROPHET/RW) si hace falta
        model = "MODEL"
        up = name.upper()
        for tag in ["ARIMA","LSTM","PROPHET","RW"]:
            if tag in up:
                model = tag
                break

        da = _direction_accuracy(df)
        rmse = _rmse(df.get("y_true"), df.get("y_pred")) if {"y_true","y_pred"} <= set(df.columns) else None

        # errores para DM (yt-yp)
        if {"y_true","y_pred"} <= set(df.columns):
            yt, yp = _align(df["y_true"], df["y_pred"])
            if len(yt) and len(yp):
                err = yt - yp
                err_series[model] = err

        rows.append({
            "model": model,
            "Direction_Accuracy(%)": round(da, 3) if da is not None else np.nan,
            "RMSE": round(rmse, 6) if rmse is not None else np.nan,
            "n": int(len(df))
        })

    metric_df = pd.DataFrame(rows).dropna(subset=["RMSE"], how="all")
    if metric_df.empty:
        print("[WARN] No hay métricas suficientes para redactar resumen.")
        return

    # 2) Ranking: 1) mayor Direction_Accuracy, 2) menor RMSE
    metric_df["__rank"] = (-metric_df["Direction_Accuracy(%)"].fillna(-1e9),
                           metric_df["RMSE"].fillna(1e9))
    metric_df = metric_df.sort_values(by=["Direction_Accuracy(%)","RMSE"], ascending=[False, True]).reset_index(drop=True)

    best = metric_df.iloc[0]
    best_model = str(best["model"])
    # segundo (si existe) para DM
    dm_text = "Sin comparación DM (un solo modelo válido)."
    if len(metric_df) >= 2:
        second = metric_df.iloc[1]
        a, b = best_model, str(second["model"])
        DM, p = (None, None)
        if a in err_series and b in err_series:
            DM, p = _dm_test(err_series[a], err_series[b], h=1, power=2)
        if DM is not None and p is not None:
            sig = "significativa" if p <= 0.10 else "no significativa"
            dm_text = f"Prueba DM entre {a} (mejor) y {b}: DM={DM:.3f}, p={p:.3f} ({sig} a 10%)."
        else:
            dm_text = f"Prueba DM entre {a} y {b}: no disponible (series insuficientes)."

    # 3) Redacción del resumen
    da_txt = f"{best['Direction_Accuracy(%)']:.2f}%" if pd.notna(best['Direction_Accuracy(%)']) else "N/D"
    rmse_txt = f"{best['RMSE']:.6f}" if pd.notna(best['RMSE']) else "N/D"

    recomendacion = (
        f"Recomendación: **{best_model}**. Motivo: mayor *Direction Accuracy* ({da_txt}) "
        f"y RMSE competitivo ({rmse_txt}). {dm_text}"
    )

    bullets = [
        "• Métrica primaria: *Direction Accuracy* (acierto direccional).",
        "• Desempate / refuerzo: *RMSE* (error cuadrático medio).",
        "• Contraste estadístico: *Diebold–Mariano (DM)* entre el mejor y el segundo.",
        "• Si DM p ≤ 0.10, la ventaja del mejor se considera estadísticamente significativa.",
    ]

    # 4) Escribir hoja 'summary_text' (no tocar otras hojas)
    try:
        with pd.ExcelWriter(xlsx, mode="a", engine="openpyxl", if_sheet_exists="replace") as wr:
            # Hoja de texto (como tabla de 1 columna)
            resumen_df = pd.DataFrame({
                "summary_text": [
                    "RESUMEN EJECUTIVO (validación + backtest)",
                    "",
                    f"Mejor modelo: {best_model}",
                    f"Direction Accuracy (mejor): {da_txt}",
                    f"RMSE (mejor): {rmse_txt}",
                    dm_text,
                    "",
                    recomendacion,
                    "",
                    "Criterios:",
                    *bullets
                ]
            })
            resumen_df.to_excel(wr, index=False, sheet_name="summary_text")

            # Hoja con ranking numérico (por transparencia)
            metric_df.drop(columns=["__rank"], errors="ignore").to_excel(
                wr, index=False, sheet_name="summary_ranking"
            )

        print(f"[INFO] Hoja 'summary_text' escrita en: {xlsx}")
    except Exception as e:
        print(f"[WARN] No se pudo escribir 'summary_text' en {xlsx}: {e}")


if __name__ == "__main__":
    main()
