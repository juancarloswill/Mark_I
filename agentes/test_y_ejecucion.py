#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agente de análisis, recomendación y validación (hold-out) Top-K
- Lee el XLSX consolidado del backtesting (hoja 'metrics' y hojas por modelo)
- Selecciona los TOP-K modelos priorizando: Hit Rate (Direction_Accuracy) ↓, luego RMSE ↑, luego Diebold-Mariano (DM)
- Reconstruye la serie (MT5) y split train/test con la misma lógica del proyecto (config.yaml → validacion)
- Reentrena cada modelo ganador en train y evalúa 1-paso rolling en test (con actualización/refit stride)
- Exporta un XLSX espejo del backtest con los resultados de validación en outputs/validacion/validacion_consolidado.xlsx
- Añade hoja 'metrics_valid' (ranking en TEST) y copia 'config_info' desde el backtest
Uso:
    python -m agentes.test_y_ejecucion --config utils/config.yaml --backtest_xlsx outputs/backtest_plots/EURUSD_backtest_consolidado.xlsx --top_k 2
"""

from __future__ import annotations

import os
import re
import sys
import yaml
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Menos ruido de TF C++
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# =========================
# Imports del proyecto (dinámicos para distintas estructuras)
# =========================

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

# Helpers EDA (con fallback)
try:
    from app.eda.eda_crispdm import _ensure_dt_index, _find_close, _resample_ohlc  # type: ignore
    _EDA_OK = True
except Exception:
    _EDA_OK = False
    def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_index()
    def _find_close(df: pd.DataFrame) -> str:
        return "Close" if "Close" in df.columns else df.columns[-1]
    def _resample_ohlc(df: pd.DataFrame, freq: str = "H", price_col: str = "Close") -> pd.DataFrame:
        return df

# MT5
_HAS_MT5 = True
try:
    from conexion.easy_Trading import Basic_funcs as _BF
except Exception as _e:
    _HAS_MT5 = False
    _BF = None
    print(f"⚠️ No se pudo importar Basic_funcs (conexion.easy_Trading): {_e}")

def _fixed_window_1d(series: pd.Series, win: int) -> np.ndarray:
    """
    Ventana univariada 1D (win,) float32 con padding a la izquierda si hace falta.
    Mantiene longitud constante para evitar retracing en el adapter LSTM.
    """
    x = series.astype("float32").to_numpy()
    if len(x) >= win:
        tail = x[-win:]
    else:
        if len(x) > 0:
            pad = np.full((win - len(x),), x[0], dtype="float32")
        else:
            pad = np.zeros((win,), dtype="float32")
        tail = np.concatenate([pad, x.astype("float32")], axis=0)
    return tail.astype("float32")


# =========================
# Utilidades
# =========================

def _safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
    return s[:80] if s else "model"

def _load_config(path: str) -> dict:
    """
    Carga YAML intentando rutas comunes si no encuentra el path directo.
    """
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates += [Path("config.yaml"), Path("utils/config.yaml"), Path("app/utils/config.yaml")]
    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No se encontró config.yaml. Probé: {', '.join(str(c) for c in candidates)}")

def _apply_data_window(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if not isinstance(cfg, dict) or not cfg:
        return df
    mode = str(cfg.get("mode", "last_n_bars")).lower()
    if mode == "date_range":
        start = cfg.get("start", None)
        end   = cfg.get("end", None)
        if start is None and end is None:
            return df
        if start is not None:
            start = pd.to_datetime(start)
        if end is not None:
            end = pd.to_datetime(end)
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

def _split_train_valid(price: pd.Series, cfg_valid: dict) -> Tuple[pd.Series, Optional[pd.Series]]:
    if not isinstance(cfg_valid, dict) or not cfg_valid:
        return price, None
    modo = str(cfg_valid.get("modo", "last_n")).lower()
    if modo == "date_range":
        start = cfg_valid.get("start", None)
        end   = cfg_valid.get("end", None)
        if start is None and end is None:
            return price, None
        if start is not None: start = pd.to_datetime(start)
        if end   is not None: end   = pd.to_datetime(end)
        if start is not None and end is not None:
            mask_valid = (price.index >= start) & (price.index <= end)
        elif start is not None:
            mask_valid = price.index >= start
        else:
            mask_valid = price.index <= end
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

def _get_series_from_mt5(config: dict) -> Tuple[pd.DataFrame, str]:
    if not _HAS_MT5:
        raise RuntimeError("MT5 no disponible: no es posible reconstruir la serie.")
    simbolo   = config.get("simbolo", "EURUSD")
    timeframe = config.get("timeframe", "H1")
    cantidad  = int(config.get("cantidad_datos", 3000))
    mt5c = config.get("mt5", {})
    bf = _BF(mt5c.get("login"), mt5c.get("password"), mt5c.get("server"), mt5c.get("path"))  # type: ignore
    print("✅ Conexión establecida con MetaTrader 5 (test_y_ejecucion)")
    df = bf.get_data_for_bt(timeframe, simbolo, cantidad)
    cols_map = {"open":"Open","high":"High","low":"Low","close":"Close","tick_volume":"TickVolume","real_volume":"Volume","time":"Date"}
    for k,v in cols_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})
    if "Date" in df.columns:
        df = df.set_index("Date")
    df = df.sort_index()
    price_col = _find_close(df)
    df = _ensure_dt_index(df)
    df = _resample_ohlc(df, freq=config.get("eda",{}).get("frecuencia_resampleo","H"), price_col=price_col)
    df = _apply_data_window(df, config.get("data_window", {}) or {})
    return df, price_col

# =========================
# Lectura XLSX (openpyxl values_only=True)
# =========================

def _read_backtest_consolidado(xlsx_path: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    from openpyxl import load_workbook
    wb = load_workbook(filename=xlsx_path, data_only=True, read_only=True)
    sheets = wb.sheetnames
    if "metrics" not in sheets:
        raise ValueError("El archivo no contiene hoja 'metrics'.")

    # metrics
    ws = wb["metrics"]
    header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    headers = [str(h) if h is not None else "" for h in header_row]
    data_iter = ws.iter_rows(min_row=2, values_only=True)
    metrics = pd.DataFrame(list(data_iter), columns=headers)

    # hojas por modelo
    per_model: Dict[str, pd.DataFrame] = {}
    for name in sheets:
        low = name.lower()
        if low in {"metrics", "config_info"}:
            continue
        ws2 = wb[name]
        header_row2 = next(ws2.iter_rows(min_row=1, max_row=1, values_only=True))
        headers2 = [str(h) if h is not None else "" for h in header_row2]
        data_iter2 = ws2.iter_rows(min_row=2, values_only=True)
        dfm = pd.DataFrame(list(data_iter2), columns=headers2)
        if "ds" in dfm.columns:
            dfm["ds"] = pd.to_datetime(dfm["ds"], errors="coerce")
            dfm = dfm.set_index("ds")
            dfm.index.name = "ds"
        per_model[name] = dfm

    return metrics, per_model

# =========================
# Selección TOP-K (HR→RMSE→DM)
# =========================

def _pick_topk_models(metrics: pd.DataFrame,
                      per_model: Dict[str, pd.DataFrame],
                      k: int,
                      prefer_col_hr: List[str] = ["Direction_Accuracy","Hit Rate","Hit_Rate","HitRate","DA","DA%"],
                      prefer_col_rmse: List[str] = ["RMSE","rmse"]) -> List[str]:
    if metrics is None or metrics.empty:
        raise ValueError("No hay métricas en la hoja 'metrics'.")

    cols = {c.lower(): c for c in metrics.columns}

    hr_col = None
    for c in prefer_col_hr:
        if c in metrics.columns: hr_col = c; break
        if c.lower() in cols:    hr_col = cols[c.lower()]; break
    if hr_col is None:
        raise ValueError("No se encontró columna de Hit Rate / Direction_Accuracy en 'metrics'.")

    rmse_col = None
    for c in prefer_col_rmse:
        if c in metrics.columns: rmse_col = c; break
        if c.lower() in cols:    rmse_col = cols[c.lower()]; break
    if rmse_col is None:
        raise ValueError("No se encontró columna RMSE en 'metrics'.")

    name_col = None
    for c in ("model","Modelo","name","Name","MODEL"):
        if c in metrics.columns:
            name_col = c; break
    if name_col is None:
        cand = [c for c in metrics.columns if metrics[c].dtype==object]
        name_col = cand[0] if cand else metrics.columns[0]

    df = metrics.copy()
    df_sorted = df.sort_values(by=[hr_col, rmse_col], ascending=[False, True]).reset_index(drop=True)
    topk = df_sorted.head(int(k)).copy()

    # DM tie-break dentro del topk (si hay empates exactos en HR y RMSE, usar MSE medio y DM con ancla)
    def _get_errors(sheet_name: str) -> Optional[pd.Series]:
        if sheet_name not in per_model:
            return None
        dfm = per_model[sheet_name].copy()
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
        import numpy as np
        e1, e2 = e1.align(e2, join="inner")
        if len(e1) < 20:
            return 0.0
        d = (np.abs(e1) ** power) - (np.abs(e2) ** power)
        d = d - d.mean()
        lag = max(h - 1, 0)
        gamma0 = np.dot(d, d) / len(d)
        s = gamma0
        for l in range(1, lag + 1):
            w = 1.0 - l / (lag + 1.0)
            gamma = np.dot(d[l:], d[:-l]) / len(d)
            s += 2.0 * w * gamma
        return float(d.mean() / math.sqrt(s / len(d))) if s > 0 else 0.0

    i = 0
    chosen = []
    while i < len(topk):
        row = topk.iloc[i]
        same = topk[(topk[hr_col]==row[hr_col]) & (topk[rmse_col]==row[rmse_col])]
        if len(same) == 1:
            chosen.append(str(row[name_col]))
            i += 1
            continue
        names = [str(r[name_col]) for _, r in same.iterrows()]
        errors_map = {n: _get_errors(n) for n in names}
        if any(v is None or v.empty for v in errors_map.values()):
            chosen.extend(names)
            i += len(same)
            continue
        mse_avg = {n: float((errors_map[n]**2).mean()) for n in names}
        names_sorted = sorted(names, key=lambda n: mse_avg[n])
        anchor = names_sorted[0]
        out_order = [anchor]
        for cand in names_sorted[1:]:
            _ = abs(dm_stat(errors_map[anchor], errors_map[cand], h=1, power=2))
            out_order.append(cand)
        chosen.extend(out_order)
        i += len(same)

    return chosen[:int(k)]

# =========================
# Señales, DM test en TEST y métricas
# =========================

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

def _dm_pvalue_test(y_true: pd.Series, yhat_a: pd.Series, yhat_b: pd.Series) -> float | None:
    import numpy as np, math
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

def _metrics_from_pred_map(pred_map_valid: Dict[str, pd.DataFrame], price_valid: pd.Series) -> pd.DataFrame:
    from math import sqrt
    rows = []
    for name, dfp in (pred_map_valid or {}).items():
        tmp = dfp.copy()
        if not isinstance(tmp.index, pd.DatetimeIndex) and 'ds' in tmp.columns:
            tmp = tmp.set_index('ds')
        if 'y_pred' in tmp.columns:
            yhat = tmp['y_pred'].astype(float)
        elif 'yhat' in tmp.columns:
            yhat = tmp['yhat'].astype(float)
        else:
            yhat = tmp.iloc[:, -1].astype(float)
        y_true = price_valid.reindex(yhat.index).ffill().dropna()
        yhat = yhat.reindex(y_true.index).astype(float)
        err = (y_true - yhat).astype(float)
        mae = err.abs().mean()
        rmse = sqrt((err**2).mean()) if len(err) else float('nan')
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

# =========================
# LSTM ventana estable (no retracing)
# =========================

def _fixed_window(series: pd.Series, win: int) -> np.ndarray:
    """
    Devuelve ventana univariada con forma (1, win, 1), dtype float32.
    Si la serie es más corta que win, hace left-pad con el primer valor.
    """
    x = series.astype("float32").to_numpy()
    if len(x) >= win:
        tail = x[-win:]
    else:
        if len(x) > 0:
            pad = np.full((win - len(x),), x[0], dtype="float32")
        else:
            pad = np.zeros((win,), dtype="float32")
        tail = np.concatenate([pad, x.astype("float32")], axis=0)
    return tail.reshape(1, win, 1).astype("float32")

# =========================
# Validación (reentrenar/actualizar y evaluar en test)
# =========================

def _roll_validate_on_test(model_name: str,
                           model_params: dict,
                           price_train: pd.Series,
                           price_test: pd.Series,
                           config: dict) -> pd.DataFrame:
    """
    Reentrena/actualiza en cada paso (o stride) y evalúa 1-paso rolling en TEST.
    Genera: y_pred, y_pred_price y signal {1,0,-1}.
    """
    target = str((config.get("bt", {}) or {}).get("target", "returns")).lower()
    freq   = str((config.get("eda", {}) or {}).get("frecuencia_resampleo", "H"))
    cfg_local = {"target": target, "freq": "H" if freq.upper().startswith("H") else "D"}
    key = model_name.strip().lower()
    cfg_local[key] = model_params or {}

    stride = int((config.get("agent", {}) or {}).get("validation", {}).get("refit_stride", 1))
    stride = max(1, stride)

    model = get_model(key, cfg_local)
    cur   = price_train.copy()
    model.fit(cur)

    preds = []
    step  = 0
    for ts in price_test.index:
        step += 1

        # 1) Predicción 1-paso
        if key == "lstm":
            win = int((model_params or {}).get("window", 64))
            w1d = _fixed_window_1d(cur, win)  # (win,) float32 -> lo que tu adapter espera
            try:
                # tu lstm_model.py hace: pd.Series(last_window) ... así que requiere 1D
                dfp = model.predict(1, last_timestamp=cur.index[-1], last_window=w1d)
                if not isinstance(dfp, pd.DataFrame):
                    # por si devuelve array
                    dfp = pd.DataFrame({"y_pred": [float(np.ravel(dfp)[0])]}, index=pd.DatetimeIndex([ts]))
            except TypeError:
                # Fallback: si el adapter admite batch, usa forma 3D estable
                X = _fixed_window(cur, win)  # (1, win, 1)
                try:
                    yhat = model.predict_on_batch(X)
                    dfp = pd.DataFrame({"y_pred": [float(np.ravel(yhat)[0])]}, index=pd.DatetimeIndex([ts]))
                except Exception:
                    dfp = model.predict(1)

        else:
            try:
                dfp = model.predict(1)
            except TypeError:
                dfp = model.predict(h=1, last_timestamp=cur.index[-1])

        if "yhat" in getattr(dfp, "columns", []):
            dfp = dfp.rename(columns={"yhat":"y_pred"})
        dfp.index = pd.DatetimeIndex([ts])

        # 2) Señal y precio previsto
        last_price = float(cur.iloc[-1])
        yhat = float(dfp["y_pred"].iloc[0])
        # --- Salvaguarda de escala si el adapter devuelve niveles y el target es returns ---
        if target == "returns":
            # si yhat luce demasiado grande para retorno 1-paso en FX, asumimos que vino en "precio"
            if not math.isfinite(yhat) or abs(yhat) > 0.2:
                # conviértelo de nivel a retorno simple aproximado
                yhat = (yhat / max(last_price, 1e-9)) - 1.0
                # y reflejarlo en el dataframe base para que el exportador reciba 'y_pred' corregido
                dfp.loc[dfp.index, "y_pred"] = yhat

        signal = _compute_signal(yhat, last_price, config, target)

        dfp["signal"] = int(signal)
        if target == "returns":
            dfp["y_pred_price"] = last_price * (1.0 + yhat)
        else:
            dfp["y_pred_price"] = float(yhat)
        
        dfp["fecha_inicio_ventana"] = cur.index.min()
        dfp["fecha_fin_ventana"] = cur.index.max()
        preds.append(dfp[["y_pred", "y_pred_price", "signal"]])

        # 3) Revelar obs real y actualizar/refit
        cur = pd.concat([cur, price_test.loc[[ts]]])

        updated = False
        for meth in ("update", "partial_fit", "fit_partial"):
            if hasattr(model, meth):
                try:
                    getattr(model, meth)(cur)
                    updated = True
                    break
                except Exception:
                    pass
        if (not updated) and (step % stride == 0):
            try:
                model.fit(cur)
            except Exception:
                pass

    out = pd.concat(preds).sort_index()
    out.index.name = "ds"
    return out

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
    print(f"💾 Validación (hold-out) exportada en {excel_path}")

# =========================
# Extra: escribir metrics_valid y copiar config_info
# =========================

def _write_extra_sheets(backtest_xlsx: str, validation_xlsx: Path, metrics_valid_df: pd.DataFrame) -> None:
    import pandas as pd
    # leer config_info del backtest si existe
    cfg_df = None
    try:
        xls = pd.ExcelFile(backtest_xlsx)
        if 'config_info' in xls.sheet_names:
            cfg_df = pd.read_excel(xls, 'config_info')
    except Exception as e:
        print(f"ℹ️ No fue posible leer 'config_info' del backtest: {e}")

    # escribir/adjuntar hojas
    try:
        with pd.ExcelWriter(validation_xlsx, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            if metrics_valid_df is not None and not metrics_valid_df.empty:
                metrics_valid_df.to_excel(writer, sheet_name='metrics_valid', index=False)
            if cfg_df is not None:
                cfg_df.to_excel(writer, sheet_name='config_info', index=False)
        print(f"💾 Hojas adicionales escritas en {validation_xlsx} (metrics_valid, config_info)")
    except FileNotFoundError:
        with pd.ExcelWriter(validation_xlsx, engine='openpyxl', mode='w') as writer:
            if metrics_valid_df is not None and not metrics_valid_df.empty:
                metrics_valid_df.to_excel(writer, sheet_name='metrics_valid', index=False)
            if cfg_df is not None:
                cfg_df.to_excel(writer, sheet_name='config_info', index=False)
        print(f"💾 Archivo de validación creado y hojas escritas en {validation_xlsx}")

# =========================
# CLI principal
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="Ruta a config.yaml")
    ap.add_argument("--backtest_xlsx", required=True, help="Ruta al XLSX consolidado del backtest")
    ap.add_argument("--outdir", default="outputs/validacion", help="Carpeta de salida para validación")
    ap.add_argument("--top_k", type=int, default=1, help="Número de mejores modelos a validar (TOP-K)")
    args = ap.parse_args()

    config = _load_config(args.config)

    # 1) Leer backtest consolidado y elegir TOP-K
    metrics, per_model = _read_backtest_consolidado(args.backtest_xlsx)
    topk_names = _pick_topk_models(metrics, per_model, k=max(1, args.top_k))
    print(f"⭐ TOP-{len(topk_names)} modelos (HR→RMSE→DM): {topk_names}")

    # 2) Reconstruir serie y split train/test
    df, price_col = _get_series_from_mt5(config)
    price = df[price_col].astype(float)
    price_train, price_test = _split_train_valid(price, config.get("validacion", {}) or {})
    if price_test is None:
        print("⚠️ No hay bloque de test definido en config['validacion']. Se detiene la validación.")
        sys.exit(0)

    # 3) Parametría desde config.modelos
    name2params = {str(m.get("name","")).strip().lower(): (m.get("params", {}) or {}) for m in (config.get("modelos") or [])}

    # 4) Reentrenar y evaluar cada modelo Top-K
    pred_map_valid: Dict[str, pd.DataFrame] = {}
    for name_raw in topk_names:
        params = name2params.get(name_raw.strip().lower(), {})
        pred_valid = _roll_validate_on_test(name_raw, params, price_train, price_test, config)
        pred_map_valid[name_raw] = pred_valid

    # 5) Exportar espejo del backtest
    symbol = re.sub(r"[^A-Za-z0-9_]+","_", str(config.get("simbolo","SYMB")).upper())
    outdir = Path(args.outdir)
    excel_path = outdir / "validacion_consolidado.xlsx"
    _export_validation(symbol, pred_map_valid, price_test, config, outdir, excel_path)

    # 6) metrics_valid en TEST y DM p-values (vs top-1)
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

    # 7) Escribir hojas extra
    _write_extra_sheets(args.backtest_xlsx, excel_path, metrics_valid)

if __name__ == "__main__":
    main()
