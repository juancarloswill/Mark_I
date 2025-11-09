#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Auditoría de resultados (con soporte de layout posicional y señales con flechas):
- Si una hoja no tiene columnas y_true/y_pred con nombre,
  intenta leer y_pred de la columna A (0) y y_true de la columna F (5),
  a partir de la fila 13 (1-indexed -> índice 12 en 0-index).

Novedades:
- --use_validacion + --validacion_dir: auto-descubre CSVs en la carpeta de validación
  (mantiene compatibilidad con --backtests).
- --debug_heads: imprime qué archivos se abren realmente y las primeras líneas.
- Normalización de señales: flechas '↑','→','↓' y textos 'BUY','HOLD','SELL','Long','Short'
  se convierten a 1, 0, -1.
"""

import argparse
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import glob
import re
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# ---------------------------------------------------------------------
# Descubrimiento de CSVs en una carpeta (p. ej., outputs/validacion)
# ---------------------------------------------------------------------
def _discover_validacion_csvs(base_dir: str | Path) -> list[str]:
    base = Path(base_dir).expanduser().resolve()
    if not base.exists():
        print(f"[WARN] Carpeta no existe: {base}")
        return []
    patrones = [
        str(base / "*ARIMA*.csv"),
        str(base / "*LSTM*.csv"),
        str(base / "*PROPHET*.csv"),
        str(base / "*RW*.csv"),
        str(base / "*.csv"),
    ]
    vistos = set()
    hallados: list[str] = []
    for pat in patrones:
        for p in glob.glob(pat):
            ap = str(Path(p).resolve())
            if ap not in vistos:
                vistos.add(ap); hallados.append(ap)
    return hallados

# ---------------------------------------------------------------------
# Utilidades de métricas/diagnósticos
# ---------------------------------------------------------------------
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

def align_on_common_index(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    idx = a.dropna().index.intersection(b.dropna().index)
    return a.loc[idx], b.loc[idx]

def diebold_mariano(ea: pd.Series, eb: pd.Series, h: int = 1, power: int = 2) -> Dict[str, float]:
    ea, eb = align_on_common_index(ea, eb)
    if len(ea) < 5 or len(eb) < 5:
        return {"DM_stat": np.nan, "p_value": np.nan}
    if power == 1:
        la = np.abs(ea); lb = np.abs(eb)
    else:
        la = ea**2; lb = eb**2
    d = la - lb
    dbar = np.mean(d)
    T = len(d)
    def autocov(x, k):
        x = x - np.mean(x)
        return np.sum(x[:T-k] * x[k:]) / T
    gamma0 = autocov(d, 0)
    var_d = gamma0
    for k in range(1, h):
        gam = autocov(d, k)
        var_d += 2 * (1 - k / (h)) * gam
    if var_d <= 0:
        return {"DM_stat": np.nan, "p_value": np.nan}
    DM_stat = dbar / np.sqrt(var_d / T)
    p_value = 2.0 * (1.0 - student_t.cdf(np.abs(DM_stat), df=max(T-1, 1)))
    return {"DM_stat": float(DM_stat), "p_value": float(p_value)}

# --------- Normalización de señales (flechas/texto → -1/0/1) ----------
_SIGNAL_MAP = {
    "↑": 1, "UP": 1, "BUY": 1, "LONG": 1, "BULL": 1, "ALCISTA": 1, "U": 1, "+1": 1,
    "→": 0, "HOLD": 0, "FLAT": 0, "NEUTRAL": 0, "STAY": 0, "S": 0, "0": 0,
    "↓": -1, "DOWN": -1, "SELL": -1, "SHORT": -1, "BEAR": -1, "BAJISTA": -1, "D": -1, "-1": -1,
}

def _coerce_signal_series(sig: pd.Series) -> pd.Series:
    """
    Convierte una serie de señales con flechas/texto/números a {-1,0,1} (float).
    Reglas:
    - Flechas: ↑→↓ → 1,0,-1
    - Texto: BUY/LONG → 1, HOLD/FLAT → 0, SELL/SHORT → -1
    - Números '1','0','-1' o 1,0,-1 se mantienen.
    - Otros valores se convierten a NaN.
    """
    s = pd.Series(sig)
    # 1) intenta numérico directo
    s_num = pd.to_numeric(s, errors="coerce")
    # 2) strings mapeados
    def _map_one(v):
        if pd.isna(v):
            return np.nan
        # si ya fue numérico (no NaN), úsalo
        try:
            vv = float(v)
            if vv in (-1.0, 0.0, 1.0):
                return vv
        except Exception:
            pass
        vs = str(v).strip().upper()
        return float(_SIGNAL_MAP.get(vs, np.nan))
    out = s_num.copy()
    # reemplaza NaN de s_num con mapeo por texto/flecha
    mask = out.isna()
    out.loc[mask] = s.loc[mask].map(_map_one)
    # asegura tipo float
    out = pd.to_numeric(out, errors="coerce")
    # limita a -1,0,1 (por si llegó 2 o -2)
    out = out.clip(-1, 1)
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
        "p_sell": float((s == -1).mean() * 100.0)
    }

# ---------------------------------------------------------------------
# Lectura robusta de CSVs (varios formatos / una-columna / json-lines / key=value)
# ---------------------------------------------------------------------
_POSSIBLE_SEPS = [",", ";", "\t", "|"]
_keyval_pattern = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([^,\s\|;]+)")

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
            kv = dict(_keyval_pattern.findall(line))
            if kv:
                rows.append(kv)
    if rows:
        return pd.DataFrame(rows)
    return None

def _normalize_common(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ds":"ds", "date":"ds", "timestamp":"ds", "time":"ds",
        "ytrue":"y_true", "yTrue":"y_true",
        "ypred":"y_pred", "yPred":"y_pred",
        "pred":"y_pred", "price_pred":"y_pred_price",
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
    # coerción numérica de pred/true si existen
    for c in ["y_true","y_pred","y_pred_price","error","abs_error"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # normaliza señal si existe (flechas/texto → -1/0/1)
    if "signal" in df.columns:
        df["signal"] = _coerce_signal_series(df["signal"])
    # también normaliza direction_* si existen
    for c in ["direction_pred","direction_true"]:
        if c in df.columns:
            df[c] = _coerce_signal_series(df[c])
    return df

def load_backtest_csv(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    df = _try_sep_read(path)
    if df is not None and df.shape[1] > 1:
        return _normalize_common(df)
    df = _try_json_per_line(path)
    if df is not None:
        return _normalize_common(df)
    df = _try_keyval_parse(path)
    if df is not None:
        return _normalize_common(df)
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

# ---------------------------------------------------------------------
# Lectura robusta de hojas de Excel (con fallback posicional)
# ---------------------------------------------------------------------
def parse_sheet_any(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    df = xl.parse(sheet)
    cols = set(df.columns)
    if {"y_true", "y_pred"}.issubset(cols):
        return df
    raw = xl.parse(sheet, header=None)
    try:
        y_pred = pd.to_numeric(raw.iloc[12:, 0], errors="coerce")
        y_true = pd.to_numeric(raw.iloc[12:, 5], errors="coerce")
        out = pd.DataFrame({"y_pred": y_pred.reset_index(drop=True),
                            "y_true": y_true.reset_index(drop=True)})
        if out[["y_pred","y_true"]].notna().sum().sum() == 0:
            return df
        return out
    except Exception:
        return df

# ---------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validacion_xlsx", required=True, help="Ruta al Excel de validación (consolidado)")
    ap.add_argument("--backtests", nargs="*", default=[], help="Lista de CSVs de backtest por modelo")
    ap.add_argument("--pip_size", type=float, default=0.0001, help="Tamaño de pip (FX) para análisis")
    ap.add_argument("--out_xlsx", default="outputs/validacion/auditoria_validacion.xlsx", help="Salida Excel de auditoría")
    ap.add_argument("--use_validacion", action="store_true",
                    help="Ignora --backtests y auto-descubre CSVs en --validacion_dir.")
    ap.add_argument("--validacion_dir",
                    default=r"C:\Users\USER\Documentos\Maestria\Mark_I\outputs\validacion",
                    help="Carpeta desde la cual auto-descubrir CSVs si usas --use_validacion.")
    ap.add_argument("--debug_heads", action="store_true",
                    help="Imprime las primeras líneas de cada CSV para depurar formatos.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_xlsx), exist_ok=True)

    print("\n[1/4] Leyendo Excel de validación:", args.validacion_xlsx)
    xl = pd.ExcelFile(args.validacion_xlsx)
    sheets = xl.sheet_names
    print("  Hojas:", sheets)

    # Selección de CSVs
    if args.use_validacion:
        print(f"[INFO] Descubriendo CSVs en: {args.validacion_dir}")
        csv_paths = _discover_validacion_csvs(args.validacion_dir)
        if not csv_paths:
            print("[WARN] No se encontraron CSVs en validacion_dir. ¿Ruta correcta?")
    else:
        csv_paths = args.backtests or []

    prefer = ["ARIMA", "LSTM", "PROPHET", "RW"]
    def _score(p: str) -> int:
        up = p.upper()
        for i, tag in enumerate(prefer):
            if tag in up:
                return i
        return len(prefer)
    csv_paths = sorted(csv_paths, key=_score)

    print("[INFO] CSVs a procesar:")
    for p in csv_paths:
        print("  -", p)
    if not csv_paths:
        raise SystemExit("No hay CSVs a procesar. Usa --use_validacion o pasa --backtests ...")

    # ---------- 1) Resumen de la validación (Excel) ----------
    summary_rows = []
    for sh in sheets:
        try:
            df = parse_sheet_any(xl, sh)
        except Exception as e:
            print(f"  [WARN] No pude leer hoja {sh}: {e}")
            continue
        cols = set(df.columns)
        info = {"sheet": sh, "n_rows": int(len(df))}
        if {"y_true","y_pred"}.issubset(cols):
            y_true = pd.Series(df["y_true"])
            y_pred = pd.Series(df["y_pred"])
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
            info.update({"note": "Hoja sin columnas y_true/y_pred (ni fallback posicional válido)"})
        summary_rows.append(info)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "sheet" in summary_df.columns:
        summary_df = summary_df.sort_values(["sheet"]).reset_index(drop=True)

    print("\n[2/4] Resumen por hoja (validación)")
    try:
        print(summary_df.fillna("").to_string(index=False))
    except Exception:
        print(summary_df.fillna(""))

    # ---------- 2) Auditoría Backtests CSV ----------
    bt_rows, sig_rows = [], []
    for path in csv_paths:
        name = os.path.basename(path)
        abspath = os.path.abspath(path)
        if args.debug_heads:
            print(f"[DEBUG] Leyendo: {abspath}")
            try:
                with open(abspath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        if i >= 2: break
                        print(f"         head[{i}]: {line.strip()[:120]}")
            except Exception as e:
                print("         (no se pudo previsualizar)", e)

        df = load_backtest_csv(path)
        if df is None:
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

    print("\n[3/4] Resumen backtests CSV")
    if not backtests_df.empty:
        try:
            print(backtests_df.fillna("").to_string(index=False))
        except Exception:
            print(backtests_df.fillna(""))
    if not signals_df.empty:
        print("\nDistribución de señales")
        try:
            print(signals_df.fillna("").to_string(index=False))
        except Exception:
            print(signals_df.fillna(""))

    # ---------- 3) DM entre modelos desde los CSV ----------
    dm_out = []
    try:
        model_errors = {}
        for path in csv_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            df = load_backtest_csv(path)
            if df is None or not {"y_true", "y_pred"}.issubset(df.columns):
                continue
            yt, yp = align_on_common_index(df["y_true"], df["y_pred"])
            e = (yt - yp).dropna()
            if len(e) > 10:
                model_errors[name] = e
        keys = list(model_errors.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                res = diebold_mariano(model_errors[a], model_errors[b], h=1, power=2)
                dm_out.append({"A": a, "B": b, **res})
    except Exception as e:
        print("[WARN] DM no calculado:", e)

    dm_df = pd.DataFrame(dm_out)

    # ---------- 4) Guardar Excel de auditoría ----------
    print("\n[4/4] Escribiendo Excel de auditoría ->", args.out_xlsx)
    with pd.ExcelWriter(args.out_xlsx, engine="xlsxwriter") as wr:
        summary_df.to_excel(wr, index=False, sheet_name="summary")
        if not backtests_df.empty:
            backtests_df.to_excel(wr, index=False, sheet_name="backtest_ranges")
        if not signals_df.empty:
            signals_df.to_excel(wr, index=False, sheet_name="signals_distribution")
        if not dm_df.empty:
            dm_df.to_excel(wr, index=False, sheet_name="dm_tests")

    print("\n✅ Auditoría completada.")
    print("   Revisa: summary (rangos/metricas), backtest_ranges, signals_distribution, dm_tests.")
    print("\nNotas:")
    print(" - Si 'summary' ahora muestra y_true/y_pred por hoja, se detectó el layout posicional (A/F desde fila 13).")
    print(" - Si 'summary' sigue sin métricas, la hoja realmente no contiene valores numéricos en esas posiciones.")
    print(" - Para auditorías más finas por ventana, exporta las series con headers desde test_y_ejecucion.py.")

if __name__ == "__main__":
    main()
