from __future__ import annotations
"""
Entry point del pipeline (EDA / normal / backtest).
Se mantiene el orden y firmas existentes; se agregan helpers
para: (1) exportador centralizado; (2) CSV con encabezados claros;
(3) multimodelo que combine Prophet/LSTM y ARIMA/SARIMA (classic_auto).
"""

import os
import sys
import argparse
import importlib
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Protocol, Callable
import re
import yaml
import numpy as np
import pandas as pd
from pathlib import Path  # [ADDED] Asegura Path disponible en todo el archivo

# Import para exportadores “ligeros” (CSV / XLSX consolidado por clase)
# Nota: estas funciones son adicionales y NO reemplazan el exportador
# centralizado (que también se invoca más abajo si existe).
from reportes.reportes_excel import (
    export_backtest_csv_per_model,
    export_backtest_excel_consolidado,
)

# =========================
# Cargas perezosas seguras
# =========================

def _import_get_model():
    """
    Intenta importar la factoría de modelos desde distintas rutas posibles
    para ser resiliente a cambios de estructura del proyecto.
    """
    for mod in ("app.utils.registry", "utils.registry", "registry"):
        try:
            m = importlib.import_module(mod)
            return getattr(m, "get_model")
        except Exception:
            continue
    raise ImportError("No se pudo importar get_model desde utils/registry.py")


def _import_bt():
    """
    Resuelve funciones críticas del backtest con rutas alternativas para
    no romper si el proyecto cambia de estructura.
    """
    def _no_excel(*_, **__):
        print("ℹ️ Exportador interno no disponible (usa el módulo centralizado).")
    def _no_plots(*_, **__):
        print("ℹ️ save_backtest_plots no disponible; se omiten gráficos.")

    module = None
    for mod in ("app.backtesting.backtest_rolling", "backtesting.backtest_rolling",
                "app.backtest_rolling", "backtest_rolling"):
        try:
            module = importlib.import_module(mod)
            break
        except Exception:
            continue
    if module is None:
        raise ImportError("No se pudo importar app.backtesting.backtest_rolling")

    evaluate_many = getattr(module, "evaluate_many", None)
    run_backtest = getattr(module, "run_backtest", None)
    run_backtest_many = getattr(module, "run_backtest_many", None)
    save_backtest_excel = getattr(module, "save_backtest_excel", _no_excel)
    save_backtest_plots = getattr(module, "save_backtest_plots", _no_plots)

    if any(x is None for x in (evaluate_many, run_backtest, run_backtest_many)):
        raise ImportError("Faltan funciones esenciales en backtest_rolling.")

    return evaluate_many, save_backtest_excel, save_backtest_plots, run_backtest, run_backtest_many

def _apply_data_window(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Aplica una ventana temporal opcional:
    - last_n_bars: recorta a las últimas N filas tras preparar el df.
    - date_range : recorta por rango [start, end]. Si end es None => hasta el último dato.
    """
    if not isinstance(cfg, dict) or not cfg:
        return df

    mode = str(cfg.get("mode", "last_n_bars")).lower()
    if mode == "date_range":
        start = cfg.get("start", None)
        end = cfg.get("end", None)
        if start is None and end is None:
            return df
        if start is not None:
            start = pd.to_datetime(start)
        if end is not None:
            end = pd.to_datetime(end)
        # Nota: si end es None, .loc[start:] ya lleva hasta el último índice disponible.
        if start is not None and end is not None:
            return df.loc[start:end]
        elif start is not None:
            return df.loc[start:]
        else:
            return df.loc[:end]
    else:
        n = int(cfg.get("n_bars", 0))
        if n > 0 and n < len(df):
            return df.iloc[-n:]
        return df


def _import_reporter() -> Optional[Callable[..., None]]:
    """
    Busca la función del exportador Excel centralizado en varios módulos posibles.
    Se aceptan nombres alternativos para compatibilidad.
    """
    candidates = [
        ("app.reportes.reportes_excel", ["exportar_backtest_excel",
                                         "guardar_backtest_excel",
                                         "generar_reporte_backtest_excel",
                                         "write_backtest_excel"]),
        ("reportes.reportes_excel",     ["exportar_backtest_excel",
                                         "guardar_backtest_excel",
                                         "generar_reporte_backtest_excel",
                                         "write_backtest_excel"]),
        ("reportes_excel",              ["exportar_backtest_excel",
                                         "guardar_backtest_excel",
                                         "generar_reporte_backtest_excel",
                                         "write_backtest_excel"]),
    ]
    for mod, names in candidates:
        try:
            m = importlib.import_module(mod)
            for n in names:
                fn = getattr(m, n, None)
                if callable(fn):
                    return fn
        except Exception:
            continue
    return None


get_model = _import_get_model()
evaluate_many, save_backtest_excel, save_backtest_plots, run_backtest, run_backtest_many = _import_bt()
_exportar_excel = _import_reporter()

# === Split temporal para train/test (hold-out) ===
def _split_train_valid(price, cfg_valid):
    """
    price: pd.Series con índice datetime y dtype numérico (ej. df['Close'])
    cfg_valid: dict con las llaves 'modo', 'n', 'start', 'end' tomadas de config['validacion']
    return: (price_train, price_valid)  # Series
    """
    import pandas as pd

    if not isinstance(cfg_valid, dict) or not cfg_valid:
        # Sin configuración -> todo es train, no hay valid
        return price, None

    modo = str(cfg_valid.get("modo", "last_n")).lower()
    if modo == "date_range":
        start = cfg_valid.get("start", None)
        end   = cfg_valid.get("end", None)
        if start is None or end is None:
            # Config incompleta -> no split
            return price, None
        start = pd.to_datetime(start)
        end   = pd.to_datetime(end)
        mask_valid = (price.index >= start) & (price.index <= end)
    else:
        # last_n por defecto
        n = int(cfg_valid.get("n", 0))
        if n <= 0 or n >= len(price):
            # n inválido -> no split
            return price, None
        mask_valid = price.index.isin(price.index[-n:])

    price_valid = price[mask_valid]
    price_train = price[~mask_valid]

    # Sanity checks para evitar splits ridículos
    if len(price_train) < 50 or len(price_valid) < 10:
        # Si queda muy pequeño, mejor no dividir
        return price, None

    return price_train, price_valid



def _safe_model_tag(tag: str) -> str:
    """
    Sanitiza un nombre de modelo para usarlo en nombres de archivo.
    - Sustituye separadores y caracteres problemáticos por "_"
    - Limita longitud para evitar rutas excesivas.
    """
    s = str(tag)
    s = s.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    s = s.replace("(", "_").replace(")", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("_")[:80]
    return s or "model"


# =========================
# Conexión MT5 (Protocol)
# =========================

_HAS_MT5 = True
try:
    from conexion.easy_Trading import Basic_funcs as _BF
except Exception as _e:
    _HAS_MT5 = False
    _BF = None
    print(f"⚠️ No se pudo importar Basic_funcs (conexion.easy_Trading): {_e}")


class MT5Client(Protocol):
    """Protocolo mínimo para el cliente MT5 usado por obtener_df_desde_mt5()."""
    def get_data_for_bt(self, timeframe: str, symbol: str, n_barras: int) -> pd.DataFrame: ...


if TYPE_CHECKING:
    BasicType = _BF if _BF is not None else MT5Client  # type: ignore
else:
    BasicType = MT5Client


# ================  EDA (fallbacks)  ================

try:
    from app.eda.eda_crispdm import ejecutar_eda, _ensure_dt_index, _find_close, _resample_ohlc  # type: ignore
    _EDA_OK = True
except Exception:
    _EDA_OK = False

    def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_index()

    def _find_close(df: pd.DataFrame) -> str:
        return "Close" if "Close" in df.columns else df.columns[-1]

    def _resample_ohlc(df: pd.DataFrame, freq: str = "D", price_col: str = "Close") -> pd.DataFrame:
        return df


# ======================  Umbrales / volatilidad  ======================

try:
    from arch import arch_model as _arch  # noqa: F401
    _HAS_GARCH = True
except Exception:
    _HAS_GARCH = False


def compute_atr_pips(df: pd.DataFrame, window: int = 14, pip_size: float = 0.0001) -> Optional[pd.Series]:
    """
    Calcula ATR en pips (si existen columnas OHLC).
    """
    for c in ("High", "Low", "Close"):
        if c not in df.columns:
            return None
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    return atr / float(pip_size)


def compute_garch_sigma_pips(price: pd.Series, pip_size: float = 0.0001) -> Optional[pd.Series]:
    """
    Volatilidad condicional GARCH en pips (si 'arch' está disponible y hay datos suficientes).
    """
    if not _HAS_GARCH:
        return None
    s = price.astype(float).dropna()
    if len(s) < 250:
        print("ℹ️ Pocos datos para GARCH (250+ recomendado).")
        return None
    try:
        from arch import arch_model
        logret = np.log(s).diff().dropna() * 100.0
        am = arch_model(logret, p=1, o=0, q=1, mean="Zero", vol="GARCH", dist="normal")
        res = am.fit(disp="off")
        sigma_pct = res.conditional_volatility
        sigma_ret = (sigma_pct / 100.0).reindex(s.index).ffill()
        return (sigma_ret * s) / float(pip_size)
    except Exception as e:
        print(f"ℹ️ GARCH no pudo calcularse: {e}")
        return None


# ======================  Datos desde MT5  ======================

def obtener_df_desde_mt5(bf: MT5Client, symbol: str, timeframe: str, n_barras: int) -> pd.DataFrame:
    """
    Descarga OHLC desde MT5 (Basic_funcs) y estandariza columnas.
    """
    df = bf.get_data_for_bt(timeframe, symbol, n_barras)
    cols_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close",
                "tick_volume": "TickVolume", "real_volume": "Volume", "time": "Date"}
    for k, v in cols_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    if "Date" in df.columns:
        df = df.set_index("Date")
    return df.sort_index()


# ======================  Helpers multi-modelo  ======================

def _collect_active_models(config: dict) -> Optional[List[dict]]:
    """
    Devuelve la lista de modelos con enabled=true (o None si no hay sección).
    """
    mlist = config.get("modelos")
    if not mlist:
        return None
    active = [m for m in mlist if m.get("enabled", True)]
    return active or None


# =========================================================
# Normalizador mínimo para CSV con encabezados consistentes
# =========================================================

def _normalize_for_csv(obj: pd.DataFrame | pd.Series | np.ndarray,
                       price: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Convierte DF/Series/ndarray en un DF con columnas: y_true, y_pred, error, residual.
    Index -> 'ds'. No asume nombres de columnas; infiere 'y_pred'.
    """
    if isinstance(obj, pd.Series):
        df = obj.to_frame()
    elif isinstance(obj, np.ndarray):
        df = pd.DataFrame(obj)
    else:
        df = obj.copy()

    if not isinstance(df.index, pd.DatetimeIndex) and "ds" in df.columns:
        df = df.set_index("ds")
    df = df.sort_index()

    df.columns = [str(c) if not isinstance(c, str) else c for c in df.columns]

    y_pred = None
    for cand in ("y_pred", "yhat", "pred", "forecast"):
        if cand in df.columns:
            y_pred = df[cand].astype(float)
            break
    if y_pred is None:
        y_pred = df.iloc[:, -1].astype(float)

    if price is not None:
        y_true = price.reindex(y_pred.index).ffill().astype(float)
    else:
        y_true = pd.Series(index=y_pred.index, data=np.nan, dtype=float, name="y_true")

    out = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    out.index.name = "ds"
    out["error"] = (out["y_true"] - out["y_pred"]).astype(float)
    out["residual"] = out["error"]
    return out


# ===================  Runners por modo  ===================

def run_modo_eda(df: pd.DataFrame, config: dict) -> None:
    """Ejecuta EDA si está disponible."""
    if 'ejecutar_eda' in globals() and callable(globals()['ejecutar_eda']):
        ejecutar_eda(df_eurusd=df, df_spy=None, cfg=config)
    else:
        print("⚠️ EDA no disponible.")


def run_modo_normal(df: pd.DataFrame, price_col: str, config: dict) -> None:
    """
    Modo normal con un único modelo configurado en `modelo:`.
    Mantiene compatibilidad con tu flujo actual.
    """
    modelo_cfg = config.get("modelo", {})
    nombre_raw = str(modelo_cfg.get("nombre", "ARIMA"))
    nombre = nombre_raw.strip().lower()
    horizonte = int(modelo_cfg.get("horizonte", 1))
    params = modelo_cfg.get("params", {})

    symbol = config.get("simbolo", "EURUSD")
    objetivo = str(modelo_cfg.get("objetivo", "retornos")).lower()
    target = "returns" if objetivo == "retornos" else "close"

    freq = config.get("eda", {}).get("frecuencia_resampleo", "D")
    price = df[price_col].astype(float)

    cfg_local = {"target": target, "freq": "H" if str(freq).upper().startswith("H") else "D", nombre: params}
    model = get_model(nombre, cfg_local)
    model.fit(price)

    if nombre == "lstm":
        win = int(params.get("window", 64))
        last_window = price.iloc[-win:]
        pred = model.predict(horizonte, last_timestamp=price.index[-1], last_window=last_window)
    else:
        pred = model.predict(horizonte)

    os.makedirs("outputs/modelos", exist_ok=True)
    pred_norm = _normalize_for_csv(pred, price=price)
    out_path = f"outputs/modelos/{symbol}_{_safe_model_tag(nombre_raw)}_forecast.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pred_norm.to_csv(out_path, index=True)
    print(f"✅ Forecast guardado en {out_path}")


def run_modo_normal_multi(df: pd.DataFrame, price_col: str, config: dict) -> None:
    """
    Modo normal para varios modelos activados en `modelos:`.
    """
    models = _collect_active_models(config)
    if not models:
        return run_modo_normal(df, price_col, config)

    price = df[price_col].astype(float)
    freq = config.get("eda", {}).get("frecuencia_resampleo", "H")
    os.makedirs("outputs/modelos", exist_ok=True)

    for mdef in models:
        name = mdef["name"]
        objetivo = (mdef.get("objetivo", "retornos")).lower()
        horizonte = int(mdef.get("horizonte", 1))
        model_key = name.strip().lower()

        cfg_local = {
            "target": "returns" if objetivo == "retornos" else "close",
            "freq": "H" if str(freq).upper().startswith("H") else "D",
            model_key: mdef.get("params", {})
        }
        model = get_model(name, cfg_local)
        model.fit(price)

        if model_key == "lstm":
            win = int(mdef.get("params", {}).get("window", 64))
            last_window = price.iloc[-win:]
            pred = model.predict(horizonte, last_timestamp=price.index[-1], last_window=last_window)
        else:
            pred = model.predict(horizonte)

        out = f"outputs/modelos/{config.get('simbolo','SYMB')}_{_safe_model_tag(name)}_forecast.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        _normalize_for_csv(pred, price=price).to_csv(out, index=True)
        print(f"✅ [{name}] Forecast guardado en {out}")


def run_modo_backtest(df: pd.DataFrame, price_col: str, config: dict) -> None:
    """
    Backtest con el engine seleccionado:
    - classic_auto → ARIMA/SARIMA con umbrales, métricas y plots.
    - model        → Prophet/LSTM (adaptadores de clase).
    """
    price = df[price_col].astype(float)
    eda_freq = config.get("eda", {}).get("frecuencia_resampleo", "H")
    cfg_bt = config.get("bt", {})

    # === NEW: Split train/valid para que el backtest use solo train ===
    cfg_valid = config.get("validacion", {}) or {}
    price_train, price_valid = _split_train_valid(price, cfg_valid)
    price_bt = price_train if price_valid is not None else price

    engine = cfg_bt.get("engine")
    modelo_nombre_raw = str(config.get("modelo", {}).get("nombre", "ARIMA"))
    modelo_nombre = modelo_nombre_raw.strip().lower()
    if engine is None:
        engine = "classic_auto" if modelo_nombre in {"arima", "sarima"} else "model"

    initial_train = int(cfg_bt.get("initial_train", 1500))
    step = int(cfg_bt.get("step", 10))
    horizon = int(cfg_bt.get("horizon", 1))
    target = str(cfg_bt.get("target", "returns")).lower()
    pip_size = float(cfg_bt.get("pip_size", 0.0001))

    if engine == "classic_auto":
        # Umbrales y auto-modelo
        threshold_mode = str(cfg_bt.get("threshold_mode", "garch")).lower()
        threshold_pips = float(cfg_bt.get("threshold_pips", 12.0))
        atr_window = int(cfg_bt.get("atr_window", 14))
        atr_k = float(cfg_bt.get("atr_k", 0.60))
        garch_k = float(cfg_bt.get("garch_k", 0.60))
        min_threshold_pips = float(cfg_bt.get("min_threshold_pips", 10.0))
        log_threshold_used = bool(cfg_bt.get("log_threshold_used", False))

        # ATR/GARCH sobre el mismo rango del backtest (train)
        df_for_atr = df.rename(columns={price_col: "Close"})
        df_for_atr_bt = df_for_atr.reindex(price_bt.index)
        atr_pips_series = compute_atr_pips(df_for_atr_bt, window=atr_window, pip_size=pip_size)
        garch_sigma_pips = compute_garch_sigma_pips(price_bt, pip_size=pip_size) if _HAS_GARCH else None

        specs = [
            {"name": "RW_RETURNS", "kind": "rw"},
            {"name": "AUTO(ARIMA/SARIMA)_RET", "kind": "auto",
             "scan": cfg_bt.get("auto", {}).get("scan", {}),
             "rescan_each_refit": cfg_bt.get("auto", {}).get("rescan_each_refit", False),
             "rescan_every_refits": cfg_bt.get("auto", {}).get("rescan_every_refits", 25)},
        ]
        print(f"[AUTO] step={step}, horizon={horizon}, target={target}, thr_mode={threshold_mode}")

        summary, preds_map = evaluate_many(
            price_bt, specs,
            initial_train=initial_train, step=step, horizon=horizon, target=target,
            pip_size=pip_size, threshold_pips=threshold_pips,
            exog_ret=None, exog_lags=None, threshold_mode=threshold_mode,
            atr_pips=atr_pips_series, atr_k=atr_k, garch_k=garch_k,
            min_threshold_pips=min_threshold_pips, garch_sigma_pips=garch_sigma_pips,
            log_threshold_used=log_threshold_used
        )

        outxlsx = cfg_bt.get("outxlsx", "outputs/evaluacion.xlsx")
        outdir_plots = cfg_bt.get("outdir_plots", "outputs/backtest_plots")
        os.makedirs(os.path.dirname(outxlsx), exist_ok=True)
        os.makedirs(outdir_plots, exist_ok=True)

        # Exportador centralizado (si existe) — pasar price_bt para alinear índices
        if _exportar_excel is not None:
            try:
                _exportar_excel(
                    path_xlsx=outxlsx,
                    price=price_bt,
                    preds_map=preds_map,
                    summary=summary,
                    config=config,
                    pip_size=pip_size,
                    threshold_pips=threshold_pips
                )
                print(f"💾 Reporte (centralizado) guardado en {outxlsx}")
            except Exception as e:
                print(f"⚠️ Exportador centralizado falló: {e}. Usando respaldo local.")
                save_backtest_excel(outxlsx, summary, preds_map)
        else:
            save_backtest_excel(outxlsx, summary, preds_map)

        try:
            save_backtest_plots(outdir_plots, price_bt, preds_map, pip_size, threshold_pips)
        except Exception as e:
            print(f"ℹ️ Plots omitidos: {e}")
        try:
            print(summary.to_string(index=False))
        except Exception:
            print(summary)
        return

    # engine == model (Prophet/LSTM)
    cfg_local = {
        "backtest": {"ventanas": initial_train, "step": step, "horizon": horizon},
        "target": target,
        "freq": "H" if eda_freq.upper().startswith("H") else "D",
        modelo_nombre: cfg_bt.get(modelo_nombre, {})
    }
    model = get_model(modelo_nombre, cfg_local)
    model.fit(price_bt)

    if modelo_nombre == "lstm":
        win = int(cfg_bt.get(modelo_nombre, {}).get("window", 64))
        last_window = price_bt.iloc[-win:]
        pred_df = run_backtest(price_bt, model, cfg_local)
    else:
        pred_df = run_backtest(price_bt, model, cfg_local)

    os.makedirs("outputs/modelos", exist_ok=True)
    out = f"outputs/modelos/{config.get('simbolo','EURUSD')}_{_safe_model_tag(modelo_nombre_raw)}_backtest.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _normalize_for_csv(pred_df, price=price_bt).to_csv(out, index=True)
    print(f"💾 Backtest (engine=model) guardado en {out}")

    if _exportar_excel is not None:
        try:
            _exportar_excel(
                path_xlsx=config.get("bt", {}).get("outxlsx", "outputs/evaluacion.xlsx"),
                price=price_bt,
                preds_map={modelo_nombre_raw: pred_df},
                summary=None,
                config=config
            )
            print("💾 Reporte (centralizado) actualizado con engine=model.")
        except Exception as e:
            print(f"ℹ️ Exportador centralizado no soportó engine=model: {e}")


def run_modo_backtest_multi(df: pd.DataFrame, price_col: str, config: dict) -> None:
    """
    Modo backtest multimodelo:
    - Ejecuta los modelos de clase (Prophet/LSTM) con engine=model y exporta sus CSV.
    - Si ARIMA/SARIMA están habilitados, corre classic_auto y agrega sus resultados.
    - Finalmente consolida todo en el exportador Excel centralizado (si existe).
    *No elimina lógica previa; solo ordena llamadas y evita imports locales.*
    """
    models = _collect_active_models(config)
    if not models:
        # Si no hay lista de modelos, usa el flujo único
        return run_modo_backtest(df, price_col, config)

    # ------------------------ Datos base ------------------------
    price = df[price_col].astype(float)

    # === NEW: Split train/valid para que el backtest use solo train ===
    cfg_valid = config.get("validacion", {}) or {}
    price_train, price_valid = _split_train_valid(price, cfg_valid)
    price_bt = price_train if price_valid is not None else price

    bt = config.get("bt", {})
    initial_train = int(bt.get("initial_train", 1500))
    step = int(bt.get("step", 10))
    horizon = int(bt.get("horizon", 1))
    target = str(bt.get("target", "returns")).lower()
    freq = config.get("eda", {}).get("frecuencia_resampleo", "H")
    pip_size = float(bt.get("pip_size", 0.0001))
    threshold_mode = str(bt.get("threshold_mode", "fixed")).lower()
    threshold_pips = float(bt.get("threshold_pips", 12.0))

    # Normalización de símbolo para nombres de archivo
    _raw_symbol = (
        config.get("simbolo")
        or config.get("symbol")
        or config.get("general", {}).get("symbol")
        or config.get("GENERAL", {}).get("symbol")
        or "SYMBOL"
    )
    symbol_safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(_raw_symbol).strip().upper())

    outdir_multi = Path(bt.get("outdir_plots", "outputs/backtest_plots"))
    outdir_multi.mkdir(parents=True, exist_ok=True)

    # ----------------- 1) Modelos de CLASE (Prophet/LSTM/ARIMA adapter) -----------------
    class_models = [m for m in models if m["name"].strip().lower() in {"prophet", "lstm", "arima"} and m.get("enabled", True)]
    cfg_global = {
        "freq": "H" if str(freq).upper().startswith("H") else "D",
        "target": target,
        "backtest": {"ventanas": initial_train, "step": step, "horizon": horizon}
    }
    pred_map_class = run_backtest_many(price_bt, class_models, cfg_global)

    # Guardar CSV por modelo de clase (inspección rápida) + contexto para métricas
    export_backtest_csv_per_model(
        symbol=symbol_safe,
        pred_map=pred_map_class,
        price=price_bt,
        outdir=outdir_multi,
        target=target,
        pip_size=pip_size,
        threshold_mode=threshold_mode,
        threshold_pips=threshold_pips,
        horizon=horizon,
        initial_train=initial_train,
    )

    # Excel consolidado por clase (opcional): hoja "metrics" + hojas por modelo
    tf = str(config.get("timeframe", "H1")).upper()
    annual = 24 * 252 if tf.startswith("H") else 252 if tf.startswith("D") else None
    per_model_params = {str(m.get("name", "")).upper(): (m.get("params", {}) or {}) for m in config.get("modelos", [])}

    excel_path = outdir_multi / f"{symbol_safe}_backtest_consolidado.xlsx"
    export_backtest_excel_consolidado(
        symbol=symbol_safe,
        pred_map=pred_map_class,
        price=price_bt,
        excel_path=excel_path,
        target=target,
        pip_size=pip_size,
        threshold_mode=threshold_mode,
        threshold_pips=threshold_pips,
        horizon=horizon,
        annualization=annual,
        per_model_params=per_model_params,
        config_info=config,
        seasonality_m=1,
        initial_train=initial_train,
    )
    print(f"💾 XLSX consolidado por clase guardado en: {excel_path}")

    # Además guarda CSV normalizados por modelo (retrocompatibilidad)
    for name, mat in pred_map_class.items():
        name_safe = _safe_model_tag(name)
        out_csv = outdir_multi / f"{config.get('simbolo','SYMB')}_{name_safe}_backtest.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        _normalize_for_csv(mat, price=price_bt).to_csv(out_csv, index=True)
        print(f"💾 [{name}] Backtest guardado en {out_csv}")

    # ----------------- 2) ¿ARIMA/SARIMA habilitados? -----------------
    has_arima = any(
        m.get("enabled", True) and m.get("name", "").strip().lower() in {"arima", "sarima"}
        for m in models
    )
    merged_map: Dict[str, Any] = dict(pred_map_class)
    summary_auto = None

    if has_arima:
        # Prepara insumos de umbral/volatilidad (sobre train)
        atr_window = int(bt.get("atr_window", 14))
        atr_k = float(bt.get("atr_k", 0.60))
        garch_k = float(bt.get("garch_k", 0.60))
        min_threshold_pips = float(bt.get("min_threshold_pips", 10.0))
        log_threshold_used = bool(bt.get("log_threshold_used", False))

        df_for_atr = df.rename(columns={price_col: "Close"})
        df_for_atr_bt = df_for_atr.reindex(price_bt.index)
        atr_pips_series = compute_atr_pips(df_for_atr_bt, window=atr_window, pip_size=pip_size)
        garch_sigma_pips = compute_garch_sigma_pips(price_bt, pip_size=pip_size) if _HAS_GARCH else None

        specs = [
            {"name": "RW_RETURNS", "kind": "rw"},
            {
                "name": "AUTO(ARIMA/SARIMA)_RET",
                "kind": "auto",
                "scan": bt.get("auto", {}).get("scan", {}),
                "rescan_each_refit": bt.get("auto", {}).get("rescan_each_refit", False),
                "rescan_every_refits": bt.get("auto", {}).get("rescan_every_refits", 25),
            },
        ]
        print(f"[AUTO] step={step}, horizon={horizon}, target={target}, thr_mode={threshold_mode}")

        summary_auto, preds_map_auto = evaluate_many(
            price_bt, specs,
            initial_train=initial_train, step=step, horizon=horizon, target=target,
            pip_size=pip_size, threshold_pips=threshold_pips,
            exog_ret=None, exog_lags=None, threshold_mode=threshold_mode,
            atr_pips=atr_pips_series, atr_k=atr_k, garch_k=garch_k,
            min_threshold_pips=min_threshold_pips, garch_sigma_pips=garch_sigma_pips,
            log_threshold_used=log_threshold_used
        )

        # Guarda CSV para el mapa ARIMA/SARIMA y RW
        for k, v in preds_map_auto.items():
            k_safe = _safe_model_tag(k)
            out = outdir_multi / f"{config.get('simbolo','SYMB')}_{k_safe}_backtest.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            _normalize_for_csv(v, price=price_bt).to_csv(out, index=True)
            print(f"💾 [{k}] Backtest guardado en {out}")

        # Mezcla resultados
        merged_map.update(preds_map_auto)

    # ----------------- 3) Exportador Excel centralizado (todo) -----------------
    if _exportar_excel is not None:
        try:
            _exportar_excel(
                path_xlsx=bt.get("outxlsx", "outputs/evaluacion.xlsx"),
                price=price_bt,
                preds_map=merged_map,
                summary=summary_auto,  # puede ser None si no hubo ARIMA/SARIMA
                config=config
            )
            print("💾 Reporte (centralizado) consolidado para modelos por clase y ARIMA/SARIMA.")
        except Exception as e:
            print(f"⚠️ Exportador centralizado falló en multi: {e}")

# =============  CLI principal  =============

def main():
    """
    CLI: selecciona modo y lee config YAML. Orquesta descarga MT5 → EDA/Normal/Backtest.
    Variables de config del nivel raíz usadas aquí:
      - simbolo, timeframe, cantidad_datos, mt5
      - eda.frecuencia_resampleo
      - modelo  (modo normal single)
      - modelos (modo normal/backtest multi)
      - bt      (parámetros de backtesting)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["eda", "normal", "backtest"], default="normal")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    simbolo   = config.get("simbolo", "EURUSD")
    timeframe = config.get("timeframe", "H1")
    cantidad  = int(config.get("cantidad_datos", 3000))
    eda_cfg   = config.get("eda", {})

    if not _HAS_MT5:
        raise SystemExit("❌ No hay conexión MT5 (conexion.easy_Trading.Basic_funcs).")

    mt5c = config.get("mt5", {})
    bf = _BF(mt5c.get("login"), mt5c.get("password"), mt5c.get("server"), mt5c.get("path"))  # type: ignore
    print("✅ Conexión establecida con MetaTrader 5")

    try:
        df = obtener_df_desde_mt5(bf, simbolo, timeframe, cantidad)
        price_col = _find_close(df)
        df = _ensure_dt_index(df)
        df = _resample_ohlc(df, freq=eda_cfg.get("frecuencia_resampleo", "H"), price_col=price_col)
        # Recorte opcional por ventana (últimas N barras o rango de fechas)
        df = _apply_data_window(df, config.get("data_window", {}) or {})


        if args.modo == "eda":
            run_modo_eda(df, config)
        elif args.modo == "normal":
            run_modo_normal_multi(df, price_col, config)
        elif args.modo == "backtest":
            run_modo_backtest_multi(df, price_col, config)
        else:
            print("ℹ️ Modo no reconocido.")
    finally:
        try:
            from MetaTrader5 import shutdown as _mt5_shutdown  # type: ignore
            _mt5_shutdown()
        except Exception:
            pass
        print("🛑 Conexión cerrada")


if __name__ == "__main__":
    main()
