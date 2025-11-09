# app/utils/auditoria.py
from __future__ import annotations
import os, glob, re, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

_POSSIBLE_SEPS = [",", ";", "\t", "|"]
_SIGNAL_MAP = {
    "↑": 1, "UP": 1, "BUY": 1, "LONG": 1, "BULL": 1, "ALCISTA": 1, "U": 1, "+1": 1,
    "→": 0, "HOLD": 0, "FLAT": 0, "NEUTRAL": 0, "STAY": 0, "S": 0, "0": 0,
    "↓": -1, "DOWN": -1, "SELL": -1, "SHORT": -1, "BEAR": -1, "BAJISTA": -1, "D": -1, "-1": -1,
}

def infer_scale(series: pd.Series) -> str:
    s = pd.Series(series).dropna().astype(float)
    if len(s) == 0: return "unknown"
    med = float(s.abs().median()); maxv = float(s.abs().max())
    if med < 0.01 and maxv < 0.2: return "returns"
    if 0.5 <= float(s.mean()) <= 5.0: return "price"
    return "unknown"

def basic_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    yt = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy()
    yp = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy()
    m = ~np.isnan(yt) & ~np.isnan(yp); yt = yt[m]; yp = yp[m]
    if len(yt) == 0: return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(np.abs(err) / np.where(np.abs(yt) < 1e-12, np.nan, np.abs(yt))) * 100.0)
    return {"n": int(len(yt)), "MAE": mae, "RMSE": rmse, "MAPE": mape}

def _coerce_signal_series(sig: pd.Series) -> pd.Series:
    s = pd.Series(sig)
    s_num = pd.to_numeric(s, errors="coerce")
    def _map_one(v):
        if pd.isna(v): return np.nan
        try:
            vv = float(v)
            if vv in (-1.0, 0.0, 1.0): return vv
        except Exception:
            pass
        vs = str(v).strip().upper()
        return float(_SIGNAL_MAP.get(vs, np.nan))
    out = s_num.copy()
    mask = out.isna()
    out.loc[mask] = s.loc[mask].map(_map_one)
    out = pd.to_numeric(out, errors="coerce").clip(-1, 1)
    return out

def describe_signals(sig: pd.Series) -> Dict[str, float]:
    s = _coerce_signal_series(sig).dropna()
    n = len(s)
    if n == 0:
        return {"n": 0, "p_buy": np.nan, "p_hold": np.nan, "p_sell": np.nan}
    return {
        "n": n,
        "p_buy": float((s == 1).mean() * 100.0),
        "p_hold": float((s == 0).mean() * 100.0),
        "p_sell": float((s == -1).mean() * 100.0),
    }

def align_on_common_index(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    idx = a.dropna().index.intersection(b.dropna().index)
    return a.loc[idx], b.loc[idx]

def diebold_mariano(ea: pd.Series, eb: pd.Series, h: int = 1, power: int = 2) -> Dict[str, float]:
    ea, eb = align_on_common_index(ea, eb)
    if len(ea) < 5 or len(eb) < 5: return {"DM_stat": np.nan, "p_value": np.nan}
    la = np.abs(ea) if power == 1 else ea**2
    lb = np.abs(eb) if power == 1 else eb**2
    d = la - lb; dbar = np.mean(d); T = len(d)
    def autocov(x, k):
        x = x - np.mean(x)
        return np.sum(x[:T-k] * x[k:]) / T
    gamma0 = autocov(d, 0); var_d = gamma0
    for k in range(1, h):
        gam = autocov(d, k); var_d += 2 * (1 - k / (h)) * gam
    if var_d <= 0: return {"DM_stat": np.nan, "p_value": np.nan}
    DM_stat = dbar / np.sqrt(var_d / T)
    from scipy.stats import t as student_t
    p_value = 2.0 (1.0 - student_t.cdf(np.abs(DM_stat), df=max(T-1, 1)))
    return {"DM_stat": float(DM_stat), "p_value": float(p_value)}

def _try_sep_read(path: str) -> pd.DataFrame | None:
    for sep in _POSSIBLE_SEPS:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1: return df
        except Exception: pass
    return None

def _try_json_per_line(path: str) -> pd.DataFrame | None:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                try: obj = json.loads(line.replace("'", '"'))
                except Exception: obj = None
            if isinstance(obj, dict): rows.append(obj)
    return pd.DataFrame(rows) if rows else None

_keyval_pattern = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([^,\s\|;]+)")
def _try_keyval_parse(path: str) -> pd.DataFrame | None:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            kv = dict(_keyval_pattern.findall(line))
            if kv: rows.append(kv)
    return pd.DataFrame(rows) if rows else None

def _normalize_common(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ds":"ds", "date":"ds", "timestamp":"ds", "time":"ds",
        "ytrue":"y_true", "yTrue":"y_true",
        "ypred":"y_pred", "yPred":"y_pred",
        "pred":"y_pred", "price_pred":"y_pred_price",
        "signal":"signal",
    }
    for c in list(df.columns):
        cc = str(c).strip()
        if cc in rename_map: df = df.rename(columns={c: rename_map[cc]})
        elif cc.lower() in rename_map: df = df.rename(columns={c: rename_map[cc.lower()]})
    for c in ["ds","date","timestamp","time"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                df = df.set_index(c); break
            except Exception: pass
    for c in ["y_true","y_pred","y_pred_price","error","abs_error"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "signal" in df.columns: df["signal"] = _coerce_signal_series(df["signal"])
    for c in ["direction_pred","direction_true"]:
        if c in df.columns: df[c] = _coerce_signal_series(df[c])
    return df

def load_backtest_csv(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path): return None
    df = _try_sep_read(path)
    if df is not None and df.shape[1] > 1: return _normalize_common(df)
    df = _try_json_per_line(path)
    if df is not None: return _normalize_common(df)
    df = _try_keyval_parse(path)
    if df is not None: return _normalize_common(df)
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
    except Exception: pass
    return None

def parse_sheet_any(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    df = xl.parse(sheet)
    if {"y_true","y_pred"}.issubset(df.columns): return df
    raw = xl.parse(sheet, header=None)
    y_pred = pd.to_numeric(raw.iloc[12:, 0], errors="coerce")
    y_true = pd.to_numeric(raw.iloc[12:, 5], errors="coerce")
    out = pd.DataFrame({"y_pred": y_pred.reset_index(drop=True),
                        "y_true": y_true.reset_index(drop=True)})
    return out

def run_auditoria(validacion_xlsx: str,
                  csv_paths: List[str],
                  out_xlsx: str = "outputs/validacion/auditoria_validacion.xlsx",
                  debug_heads: bool = False) -> None:
    xl = pd.ExcelFile(validacion_xlsx)
    sheets = xl.sheet_names
    # 1) resumen por hoja
    summary_rows = []
    for sh in sheets:
        df = parse_sheet_any(xl, sh)
        info = {"sheet": sh, "n_rows": int(len(df))}
        if {"y_true","y_pred"}.issubset(df.columns):
            y_true = pd.Series(df["y_true"]); y_pred = pd.Series(df["y_pred"])
            info.update({
                "y_true_min": float(pd.to_numeric(y_true, errors="coerce").min()),
                "y_true_max": float(pd.to_numeric(y_true, errors="coerce").max()),
                "y_pred_min": float(pd.to_numeric(y_pred, errors="coerce").min()),
                "y_pred_max": float(pd.to_numeric(y_pred, errors="coerce").max()),
                "scale_true": infer_scale(y_true),
                "scale_pred": infer_scale(y_pred),
                **basic_metrics(y_true, y_pred),
            })
        else:
            info.update({"note": "Hoja sin columnas y_true/y_pred"})
        summary_rows.append(info)
    summary_df = pd.DataFrame(summary_rows).sort_values("sheet").reset_index(drop=True)

    # 2) backtests
    bt_rows, sig_rows = [], []
    for path in csv_paths:
        name = os.path.basename(path)
        if debug_heads and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        if i>=2: break
                        print(f"[DEBUG:{name}] {line.strip()[:120]}")
            except Exception: pass
        df = load_backtest_csv(path)
        if df is None: continue
        cols = set(df.columns)
        row = {"file": name, "n_rows": int(len(df))}
        ycols = [c for c in ["y_true","y_pred","y_pred_price"] if c in cols]
        scols = [c for c in ["signal","direction_pred","direction_true"] if c in cols]
        row["y_cols"] = ",".join(ycols); row["signal_cols"] = ",".join(scols)
        if {"y_true","y_pred"}.issubset(cols):
            row.update({f"pred_{k}": v for k, v in basic_metrics(df["y_true"], df["y_pred"]).items()})
            row["scale_true"] = infer_scale(df["y_true"]); row["scale_pred"] = infer_scale(df["y_pred"])
        if "y_true" in cols and "y_pred_price" in cols:
            row.update({f"pred_price_{k}": v for k, v in basic_metrics(df["y_true"], df["y_pred_price"]).items()})
        bt_rows.append(row)
        for sc in scols:
            sig_rows.append({"file": name, "signal_col": sc, **describe_signals(df[sc])})
    backtests_df = pd.DataFrame(bt_rows).sort_values("file").reset_index(drop=True) if bt_rows else pd.DataFrame()
    signals_df = pd.DataFrame(sig_rows).sort_values(["file","signal_col"]).reset_index(drop=True) if sig_rows else pd.DataFrame()

    # 3) DM tests (si hay datos comparables)
    dm_out = []
    model_errors = {}
    for path in csv_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        df = load_backtest_csv(path)
        if df is None or not {"y_true","y_pred"}.issubset(df.columns): continue
        yt, yp = align_on_common_index(df["y_true"], df["y_pred"])
        e = (yt - yp).dropna()
        if len(e) > 10: model_errors[name] = e
    keys = list(model_errors.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a, b = keys[i], keys[j]
            res = diebold_mariano(model_errors[a], model_errors[b], h=1, power=2)
            dm_out.append({"A": a, "B": b, **res})
    dm_df = pd.DataFrame(dm_out)

    # 4) escribir Excel
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as wr:
        summary_df.to_excel(wr, index=False, sheet_name="summary")
        if not backtests_df.empty: backtests_df.to_excel(wr, index=False, sheet_name="backtest_ranges")
        if not signals_df.empty:  signals_df.to_excel(wr, index=False, sheet_name="signals_distribution")
        if not dm_df.empty:       dm_df.to_excel(wr, index=False, sheet_name="dm_tests")
