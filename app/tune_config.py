# -*- coding: utf-8 -*-
"""
Tuner por grid (corrige uso de datos fijos):
- Lee SIEMPRE simbolo, timeframe, cantidad_datos y credenciales MT5 desde el YAML.
- Usa conexion.easy_Trading.Basic_funcs (o easy_Trading.Basic_funcs) para conectarse a MT5 y traer OHLC.
- Split TRAIN/TEST desde YAML (split: last_n/ratio/date o fallback validacion.last_n).
- Calcula localmente ATR en pips y un proxy de sigma en pips (para 'garch').
- Lanza evaluate_many (engine classic_auto) o registry (engine model).
- Log JSONL/CSV + YAML optimizado con tuning_summary.
"""

from __future__ import annotations
import os, json
from itertools import product
from typing import Dict, Any, Iterable, Tuple, Optional
import numpy as np
import pandas as pd
import yaml
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ValueWarning)

# --------- Import de Basic_funcs desde conexion.easy_Trading ---------
Basic_funcs = None
try:
    from conexion.easy_Trading import Basic_funcs as _BF
    Basic_funcs = _BF
except Exception:
    try:
        from conexion.easy_Trading import Basic_funcs as _BF2
        Basic_funcs = _BF2
    except Exception as e:
        raise ImportError("No se pudo importar Basic_funcs. Asegúrate de tener conexion/easy_Trading.py o easy_Trading.py en el PYTHONPATH") from e

# evaluate_many del backtesting
try:
    from app.backtesting.backtest_rolling import evaluate_many
except Exception:
    from app.backtesting.backtest_rolling import evaluate_many

# Opcional (engine model)
try:
    from app.utils.registry import get_model, run_backtest, build_backtest_frame, compute_generic_metrics
except Exception:
    try:
        from utils.registry import get_model, run_backtest, build_backtest_frame, compute_generic_metrics
    except Exception:
        get_model = run_backtest = build_backtest_frame = compute_generic_metrics = None

# --------- Helpers de logging ---------
def _write_jsonl(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _log_txt(path_txt: str, split: str, params: dict, metrics: dict, objective: str = None):
    os.makedirs(os.path.dirname(path_txt), exist_ok=True)
    obj_val = metrics.get(objective) if objective else None
    pf = metrics.get('ProfitFactor')
    wr = metrics.get('HitRate_%') or metrics.get('Directional_Accuracy_%')
    dd = metrics.get('MaxDD_pips') or metrics.get('MaxDD')
    print(f"[{split}] params={params} | PF={pf} | WinRate={wr} | MaxDD={dd} | objective({objective})={obj_val}")
    with open(path_txt, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'split': split, 'params': params, 'metrics': metrics}, ensure_ascii=False) + '\n')

def _iter_param_grid(param_dict: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    keys, values, fixed = [], [], {}
    for k, v in (param_dict or {}).items():
        if isinstance(v, list): keys.append(k); values.append(v)
        else: fixed[k] = v
    if not keys:
        yield dict(fixed); return
    from itertools import product
    for combo in product(*values):
        cand = dict(fixed); cand.update({k: c for k, c in zip(keys, combo)}); yield cand

def _collect_sweep_candidates(cfg: dict, engine: str, model_name: str | None = None) -> Dict[str, Any]:
    sweep, bt = {}, (cfg.get("bt", {}) or {})
    if engine == "classic_auto":
        for k, v in bt.items():
            if isinstance(v, list): sweep[k] = v
        if isinstance(bt.get("auto"), dict):
            for k, v in bt["auto"].items():
                if isinstance(v, list): sweep[f"auto.{k}"] = v
        return sweep
    if model_name and isinstance(bt.get(model_name), dict):
        for k, v in bt[model_name].items():
            if isinstance(v, list): sweep[k] = v
    for m in (cfg.get("modelos", []) or []):
        if str(m.get("name","")).strip().lower() == str(model_name or "").lower():
            for k, v in (m.get("params") or {}).items():
                if isinstance(v, list): sweep[k] = v
    return sweep

def _score_of(metrics: dict, objective: str, maximize: bool) -> float:
    try: v = float(metrics.get(objective))
    except Exception: v = float("-inf") if maximize else float("inf")
    return v

def _num_from_cfg(v, default):
    """
    Devuelve un número a partir de un valor de config:
    - Si `v` es lista, intenta el primer elemento convertible a float.
    - Si `v` es None o no convertible, retorna `default`.
    """
    if isinstance(v, list):
        for item in v:
            try:
                return float(item)
            except Exception:
                continue
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

# --------- Split y carga con Basic_funcs ---------
def _apply_split(series: pd.Series, cfg: dict) -> Tuple[pd.Series, Optional[pd.Series]]:
    sp = cfg.get("split", {}) or {}
    mode = str(sp.get("mode", "")).lower()
    if mode == "last_n":
        n = int(sp.get("n", 0))
        return (series.iloc[:-n], series.iloc[-n:]) if (n and n < len(series)) else (series, None)
    if mode == "ratio":
        r = float(sp.get("train_ratio", 0.8)); cut = max(1, int(len(series)*r))
        return series.iloc[:cut], (series.iloc[cut:] if cut < len(series) else None)
    if mode == "date":
        d = str(sp.get("date",""))
        if d:
            mask = series.index < pd.to_datetime(d)
            tr, te = series[mask], series[~mask]
            return (tr if len(tr) else series), (te if len(te) else None)
    valid = cfg.get("validacion", {}) or {}
    if str(valid.get("modo","none")).lower() == "last_n":
        n = int(valid.get("n", 0))
        return (series.iloc[:-n], series.iloc[-n:]) if (n and n < len(series)) else (series, None)
    return series, None

def _load_ohlc_from_mt5(cfg: dict) -> pd.DataFrame:
    # Lee credenciales de cfg['mt5']
    mt5_cfg = cfg.get("mt5", {}) or {}
    login = mt5_cfg.get("login"); password = mt5_cfg.get("password")
    server = mt5_cfg.get("server"); path = mt5_cfg.get("path")
    if any(v is None for v in (login, password, server)):
        raise RuntimeError("Faltan credenciales MT5 en config: mt5.login/password/server")

    simbolo = cfg.get("simbolo", "EURUSD")
    timeframe = cfg.get("timeframe", "D1")
    cantidad = int(cfg.get("cantidad_datos", 3000))

    # Conexión MT5 a través de Basic_funcs (una sola sesión)
    bf = Basic_funcs(login=login, password=password, server=server, path=path)
    df = bf.get_data_for_bt(timeframe=timeframe, symbol=simbolo, count=cantidad)
    return df

# --------- Cálculos locales de umbrales ---------
def compute_atr_pips_local(df_ohlc: pd.DataFrame, window: int, pip_size: float) -> pd.Series:
    for c in ("High","Low","Close"):
        if c not in df_ohlc.columns: return pd.Series([], dtype=float)
    high, low, close = df_ohlc["High"].astype(float), df_ohlc["Low"].astype(float), df_ohlc["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/window, adjust=False).mean()
    return (atr / float(pip_size)).rename("ATR_pips")

def compute_sigma_pips_proxy(price: pd.Series, pip_size: float, window: int = 20) -> pd.Series:
    ret = np.log(price.astype(float)).diff()
    sigma = ret.rolling(window).std()
    price = price.reindex(sigma.index)
    delta = (sigma * price).abs()
    return (delta / float(pip_size)).rename("SIGMA_pips")

def _load_price_and_helpers(cfg: dict):
    df = _load_ohlc_from_mt5(cfg)  # <--- MT5 con credenciales del YAML
    price_col = "Close"
    price_series = df[price_col].dropna()

    train, test = _apply_split(price_series, cfg)
    pip_size = float((cfg.get("bt", {}) or {}).get("pip_size", 0.0001))

    bt = cfg.get("bt", {}) or {}
    atr_window = int(bt.get("atr_window", 14))
    atr_pips_train = compute_atr_pips_local(df.loc[train.index], atr_window, pip_size)
    sigma_pips_train = compute_sigma_pips_proxy(train, pip_size, window=int(_num_from_cfg(bt.get("garch_window", 20), 20)))

    atr_pips_test = compute_atr_pips_local(df.loc[test.index], atr_window, pip_size) if test is not None else None
    sigma_pips_test = compute_sigma_pips_proxy(test, pip_size, window=int(_num_from_cfg(bt.get("garch_window", 20), 20))) if test is not None else None

    return df, train, test, pip_size, price_col, atr_pips_train, sigma_pips_train, atr_pips_test, sigma_pips_test

# --------- Tuner principal ---------
def main_tuning(config_in: str = "utils/config_2.yaml",
                config_out: str = "utils/config_optimizado_2.yaml") -> None:
    with open(config_in, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    bt = cfg.get("bt", {}) or {}
    engine = str(bt.get("engine","classic_auto")).lower()
    modelo_nombre = str((cfg.get("modelo",{}) or {}).get("nombre","ARIMA")).strip()

    log_dir = (cfg.get("tuning",{}) or {}).get("log_dir","outputs/tuning")
    os.makedirs(log_dir, exist_ok=True)
    log_jsonl = os.path.join(log_dir, "grid_results.jsonl")
    log_csv   = os.path.join(log_dir, "grid_results.csv")
    log_txt   = os.path.join(log_dir, "grid_results.txt")
    for p in (log_jsonl, log_csv, log_txt):
        if os.path.exists(p): os.remove(p)

    objective = str((bt.get("optimize") or {}).get("objective","ProfitFactor"))
    maximize  = bool((bt.get("optimize") or {}).get("maximize", True))

    df, price_bt, price_valid, pip_size, price_col, atr_bt, sig_bt, atr_v, sig_v = _load_price_and_helpers(cfg)

    initial_train = int(bt.get("initial_train", max(100, int(0.7*len(price_bt)))))
    step = int(bt.get("step",5)); horizon = int(bt.get("horizon",1))
    target = str(bt.get("target","returns")).lower()

    threshold_mode     = str(bt.get("threshold_mode","garch")).lower()
    threshold_pips     = _num_from_cfg(bt.get("threshold_pips", 12.0), 12.0)
    atr_k              = _num_from_cfg(bt.get("atr_k", 0.60), 0.60)
    garch_k            = _num_from_cfg(bt.get("garch_k", 0.60), 0.60)
    min_threshold_pips = _num_from_cfg(bt.get("min_threshold_pips", 10.0), 10.0)
    log_threshold_used = bool(bt.get("log_threshold_used", False))

    grid = _collect_sweep_candidates(cfg, engine=('model' if engine=='model' else 'classic_auto'), model_name=modelo_nombre) or {}

    filas_csv, mejor = [], None

    if engine == "classic_auto":
        specs = [
            {"name":"RW_RETURNS","kind":"rw"},
            {"name":"AUTO(ARIMA/SARIMA)_RET","kind":"auto",
             "scan": (bt.get("auto",{}) or {}).get("scan",{}),
             "rescan_each_refit": (bt.get("auto",{}) or {}).get("rescan_each_refit", False),
             "rescan_every_refits": (bt.get("auto",{}) or {}).get("rescan_every_refits", 25)},
        ]

        for params in _iter_param_grid(grid):
            bt_local = dict(bt)
            for k, v in params.items():
                if "." in k:
                    head, tail = k.split(".",1)
                    if isinstance(bt_local.get(head), dict):
                        sub = dict(bt_local[head]); sub[tail]=v; bt_local[head]=sub
                else:
                    bt_local[k]=v

            thr_mode = str(bt_local.get("threshold_mode", threshold_mode)).lower()
            thr_pips = _num_from_cfg(bt_local.get("threshold_pips", threshold_pips), threshold_pips)
            atr_k_   = _num_from_cfg(bt_local.get("atr_k", atr_k), atr_k)
            garch_k_ = _num_from_cfg(bt_local.get("garch_k", garch_k), garch_k)
            min_thr  = _num_from_cfg(bt_local.get("min_threshold_pips", min_threshold_pips), min_threshold_pips)
            log_thr  = bool(bt_local.get("log_threshold_used", log_threshold_used))

            # ---- TRAIN ----
            summary_tr, _ = evaluate_many(
                price_bt, specs, initial_train=initial_train, step=step, horizon=horizon, target=target,
                pip_size=pip_size, threshold_pips=thr_pips, exog_ret=None, exog_lags=None,
                threshold_mode=thr_mode, atr_pips=atr_bt, atr_k=atr_k_, garch_k=garch_k_,
                min_threshold_pips=min_thr, garch_sigma_pips=sig_bt, log_threshold_used=log_thr
            )
            row_tr = summary_tr.loc[summary_tr["Modelo"].astype(str).str.contains("AUTO", na=False)].iloc[0] \
                     if summary_tr["Modelo"].astype(str).str.contains("AUTO", na=False).any() else summary_tr.iloc[0]
            met_tr = row_tr.to_dict()
            _write_jsonl(log_jsonl, {"split":"train","engine":"classic_auto","params":params,"metrics":met_tr})
            _log_txt(log_txt, 'train', params, met_tr, objective)
            filas_csv.append({"split":"train", **{f"param.{k}":v for k,v in params.items()}, **{f"metric.{k}":v for k,v in met_tr.items()}})

            # ---- TEST ----
            if price_valid is not None and len(price_valid) > (initial_train + horizon + step):
                summary_va, _ = evaluate_many(
                    price_valid, specs, initial_train=min(initial_train, max(50, int(0.7*len(price_valid)))),
                    step=step, horizon=horizon, target=target, pip_size=pip_size, threshold_pips=thr_pips,
                    exog_ret=None, exog_lags=None, threshold_mode=thr_mode, atr_pips=atr_v, atr_k=atr_k_,
                    garch_k=garch_k_, min_threshold_pips=min_thr, garch_sigma_pips=sig_v, log_threshold_used=log_thr
                )
                row_va = summary_va.loc[summary_va["Modelo"].astype(str).str.contains("AUTO", na=False)].iloc[0] \
                         if summary_va["Modelo"].astype(str).str.contains("AUTO", na=False).any() else summary_va.iloc[0]
                met_va = row_va.to_dict()
                _write_jsonl(log_jsonl, {"split":"test","engine":"classic_auto","params":params,"metrics":met_va})
                _log_txt(log_txt, 'test', params, met_va, objective)
                filas_csv.append({"split":"test", **{f"param.{k}":v for k,v in params.items()}, **{f"metric.{k}":v for k,v in met_va.items()}})
                chosen = met_va if (objective in met_va) else met_tr
            else:
                chosen = met_tr

            s = _score_of(chosen, objective, True if maximize else False)
            if mejor is None:
                mejor = (s, dict(params), dict(chosen), "test" if chosen is not met_tr else "train")
            else:
                cur = mejor[0]
                if (s>cur and maximize) or (s<cur and not maximize):
                    mejor = (s, dict(params), dict(chosen), "test" if chosen is not met_tr else "train")

    else:
        if not all([get_model, run_backtest, build_backtest_frame, compute_generic_metrics]):
            raise RuntimeError("Engine 'model' requiere registry.get_model/run_backtest/build_backtest_frame/compute_generic_metrics")

        grid = _collect_sweep_candidates(cfg, engine="model", model_name=modelo_nombre) or {}
        freq = str((cfg.get("eda",{}) or {}).get("frecuencia_resampleo","D")).upper()

        for params in _iter_param_grid(grid):
            cfg_local = {"backtest": {"ventanas": initial_train, "step": step, "horizon": horizon},
                         "target": target, "freq": freq, modelo_nombre: params}
            # ---- TRAIN ----
            model = get_model(modelo_nombre, cfg_local); model.fit(price_bt)
            pred_df = run_backtest(price_bt, model, cfg_local)
            df_std = build_backtest_frame(price_bt, pred_df.iloc[:,-1], horizon, modelo_nombre, freq)
            met_tr = compute_generic_metrics(df_std, pip_size=pip_size)
            _write_jsonl(log_jsonl, {"split":"train","engine":"model","model":modelo_nombre,"params":params,"metrics":met_tr})
            _log_txt(log_txt, 'train', params, met_tr, objective)
            filas_csv.append({"split":"train", **{f"param.{k}":v for k,v in params.items()}, **{f"metric.{k}":v for k,v in met_tr.items()}})

            # ---- TEST ----
            if price_valid is not None and len(price_valid) > (initial_train + horizon + step):
                model_v = get_model(modelo_nombre, cfg_local); model_v.fit(price_valid.iloc[:-horizon])
                pred_v = run_backtest(price_valid, model_v, cfg_local)
                df_std_v = build_backtest_frame(price_valid, pred_v.iloc[:,-1], horizon, modelo_nombre, freq)
                met_va = compute_generic_metrics(df_std_v, pip_size=pip_size)
                _write_jsonl(log_jsonl, {"split":"test","engine":"model","model":modelo_nombre,"params":params,"metrics":met_va})
                _log_txt(log_txt, 'test', params, met_va, objective)
                filas_csv.append({"split":"test", **{f"param.{k}":v for k,v in params.items()}, **{f"metric.{k}":v for k,v in met_va.items()}})
                chosen = met_va if (objective in met_va) else met_tr
            else:
                chosen = met_tr

            s = _score_of(chosen, objective, True if maximize else False)
            if mejor is None:
                mejor = (s, dict(params), dict(chosen), "test" if chosen is not met_tr else "train")
            else:
                cur = mejor[0]
                if (s>cur and maximize) or (s<cur and not maximize):
                    mejor = (s, dict(params), dict(chosen), "test" if chosen is not met_tr else "train")

    # Guardar CSV
    if filas_csv:
        pd.DataFrame(filas_csv).to_csv(log_csv, index=False)

    # YAML optimizado + resumen
    new_cfg = dict(cfg)
    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "objective": objective, "maximize": maximize,
        "best": {"params": (mejor[1] if mejor else {}), "metrics": (mejor[2] if mejor else {}), "split": (mejor[3] if mejor else "train")},
        "data": {"simbolo": cfg.get("simbolo"), "timeframe": cfg.get("timeframe")},
        "logs": {"jsonl": log_jsonl, "csv": log_csv}
    }
    if mejor:
        if engine == "classic_auto":
            bt_new = dict(new_cfg.get("bt", {}) or {})
            for k, v in mejor[1].items():
                if "." in k:
                    head, tail = k.split(".",1); sub = dict(bt_new.get(head, {}) or {}); sub[tail]=v; bt_new[head]=sub
                else:
                    bt_new[k]=v
            new_cfg["bt"] = bt_new
        else:
            bt_new = dict(new_cfg.get("bt", {}) or {})
            blk = dict(bt_new.get(modelo_nombre, {}) or {}); blk.update(mejor[1]); bt_new[modelo_nombre]=blk; new_cfg["bt"]=bt_new
    new_cfg["tuning_summary"] = summary

    os.makedirs(os.path.dirname(config_out) or ".", exist_ok=True)
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"✅ Tuning finalizado. Mejor combinación: {mejor[1] if mejor else {}}")
    print(f"📈 Métricas mejor combinación ({summary['best']['split']}): {mejor[2] if mejor else {}}")
    print(f"💾 Config optimizada escrita en: {config_out}")
    print(f"💾 Logs: {log_jsonl} | {log_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_in", default="utils/config_2.yaml")
    ap.add_argument("--config_out", default="utils/config_optimizado_2.yaml")
    args = ap.parse_args()
    main_tuning(args.config_in, args.config_out)
