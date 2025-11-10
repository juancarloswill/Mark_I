# agentes/execution_agent.py
# -*- coding: utf-8 -*-
"""
Agente de Ejecución (MT5) con soporte de zona horaria y sesiones
----------------------------------------------------------------
- Elige señal desde archivo de señal o desde el consolidado (mejor modelo).
- Calcula lotaje por % de riesgo y distancia al SL (usa Basic_funcs.calculate_position_size).
- Construye Entry/SL/TP según reglas si la señal no los trae.
- Valida ventanas de operación por sesiones/días y spread máximo.
- Ejecuta la orden y (opcional) mueve SL a Breakeven.

Uso:
    python -m agentes.execution_agent --config utils/execution_windows.yaml --run once
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd

# --- TZ support cross-version (silence Pylance) ---
import sys as _sys
from datetime import datetime

if _sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
else:  # Py<3.9
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore
    except ImportError:
        ZoneInfo = None  # fallback: sin TZ específica

DAYMAP = {"Sun": 6, "Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5}


# IMPORTANTE: ajusta el import a tu estructura real del proyecto.
try:
    from conexion.easy_Trading import Basic_funcs
except Exception:
    # Fallback por si el usuario coloca el archivo junto a este módulo
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from easy_Trading import Basic_funcs  # type: ignore
    except Exception as e:
        raise ImportError(
            "No se pudo importar Basic_funcs. Asegúrate de que 'conexion/easy_Trading.py' "
            "está en tu PYTHONPATH o copia easy_Trading.py junto a este archivo."
        ) from e


# -------------------------------
# Datos de señal
# -------------------------------
@dataclass
class Signal:
    symbol: str
    side: str          # 'buy' | 'sell'
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    confidence: Optional[float] = None
    model: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


# -------------------------------
# Helpers de lectura de archivos
# -------------------------------
def _read_last_row_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(f"El CSV está vacío: {path}")
    return df.iloc[-1]


def _read_last_row_xlsx(path: str, sheet: Optional[str] = None) -> pd.Series:
    df = pd.read_excel(path, sheet_name=sheet)
    if isinstance(df, dict):
        if sheet and sheet in df:
            dd = df[sheet]
        else:
            dd = next((v for v in df.values() if isinstance(v, pd.DataFrame) and len(v) > 0), None)
            if dd is None:
                raise ValueError(f"No encontré datos en {path}")
        df = dd
    if len(df) == 0:
        raise ValueError(f"El Excel/hoja está vacío: {path} (sheet={sheet})")
    return df.iloc[-1]


def _pick_best_model_from_metrics(metrics_df: pd.DataFrame, criterion: str, mapping: dict,
                                  model_column_hint: Optional[str] = None,
                                  metric_name_column_hint: Optional[str] = None,
                                  wide_format: Optional[bool] = None) -> str:
    """
    Soporta dos formatos de la hoja 'metrics':
      1) LONG: cada fila es un modelo y hay una columna 'Model'/'Modelo' (o la indicada por model_column_hint).
               Se ordena por 'criterion' y se toma la fila top-1.
      2) WIDE: cada columna es un modelo (p. ej., 'ARIMA','LSTM','PROPHET') y hay una columna 'Metric'/'Métrica'
               con los nombres de cada métrica en filas. Se busca la fila == criterion y se escoge el mayor por columnas.

    mapping: dict con claves = nombre de modelo en la hoja y valor = ruta CSV (solo se usa para verificar columnas);
             las columnas candidatas serán las keys del mapping.
    """
    df = metrics_df.copy()

    # Hints/heurísticas
    model_col_candidates = [c for c in [model_column_hint, "Model", "Modelo", "model", "modelo"] if c]
    metric_name_col_candidates = [c for c in [metric_name_column_hint, "Metric", "Métrica", "Metrica", "metric"] if c]

    # Si wide_format está fijado en YAML, respetar
    if wide_format is True:
        # formato WIDE
        # buscar columna con nombre de métrica
        mcol = next((c for c in metric_name_col_candidates if c in df.columns), None)
        if not mcol:
            raise ValueError("Formato WIDE: no encontré columna con los nombres de métricas (ej. 'Metric'/'Métrica').")
        row = df[df[mcol].astype(str).str.strip().str.lower() == criterion.strip().lower()]
        if row.empty:
            raise ValueError(f"Formato WIDE: no encontré una fila con la métrica '{criterion}'.")
        row = row.iloc[0]
        # columnas candidatas = modelos en mapping
        candidates = [k for k in mapping.keys() if k in df.columns]
        if not candidates:
            # si no hay candidatos, intenta usar columnas no mcol
            candidates = [c for c in df.columns if c != mcol]
        sub = row[candidates].astype(float)
        best_model = sub.idxmax()
        return str(best_model)

    # Si long o auto: intentar LONG primero
    model_col = next((c for c in model_col_candidates if c in df.columns), None)
    if model_col and (criterion in df.columns):
        # LONG
        best_row = df.sort_values(criterion, ascending=False).iloc[0]
        best_model = str(best_row[model_col]).strip()
        if best_model:
            return best_model

    # Intentar WIDE automáticamente si LONG no funcionó
    mcol = next((c for c in metric_name_col_candidates if c in df.columns), None)
    # verificar si hay columnas de modelos presentes
    model_cols_present = [k for k in mapping.keys() if k in df.columns]
    if mcol and model_cols_present:
        row = df[df[mcol].astype(str).str.strip().str.lower() == criterion.strip().lower()]
        if row.empty:
            raise ValueError(f"No encontré la métrica '{criterion}' en la columna '{mcol}'.")
        row = row.iloc[0]
        sub = row[model_cols_present].astype(float)
        best_model = sub.idxmax()
        return str(best_model)

    # Si llega aquí, no se pudo inferir
    raise ValueError("No encontré columnas adecuadas para identificar el mejor modelo (ni LONG ni WIDE).")


def load_signal_from_files(conf: dict) -> Signal:
    """
    A) 'signal_file' (CSV/XLSX) con columnas: signal/buy/sell o direction_pred (1/-1), entry/sl/tp opcionales.
    B) 'best_model_metrics' (XLSX consolidado) -> escoge mejor modelo (por 'criterion') y lee la última fila
       del CSV mapeado a ese modelo.
    """
    # A) Archivo directo de señal
    signal_file = conf.get("signal_file", "")
    if signal_file and os.path.exists(signal_file):
        last = _read_last_row_csv(signal_file) if signal_file.lower().endswith(".csv") \
               else _read_last_row_xlsx(signal_file, conf.get("signal_sheet"))
        cols = {c.lower(): c for c in last.index}
        symbol = last[cols.get("symbol")] if "symbol" in cols else conf["symbol"]
        side = str(last[cols.get("signal", "signal")]).lower() if "signal" in cols else None
        entry = float(last[cols["entry"]]) if "entry" in cols and pd.notna(last[cols["entry"]]) else None
        sl    = float(last[cols["sl"]])    if "sl" in cols and pd.notna(last[cols["sl"]]) else None
        tp    = float(last[cols["tp"]])    if "tp" in cols and pd.notna(last[cols["tp"]]) else None
        model = last[cols["model"]] if "model" in cols else None
        confidence = float(last[cols["confidence"]]) if "confidence" in cols and pd.notna(last[cols["confidence"]]) else None
        if side not in {"buy", "sell"}:
            if "direction_pred" in cols:
                dp = float(last[cols["direction_pred"]])
                side = "buy" if dp > 0 else "sell"
        if side not in {"buy", "sell"}:
            raise ValueError(f"No se pudo inferir 'side' (signal/buy/sell) desde {signal_file}")
        return Signal(symbol=symbol, side=side, entry=entry, sl=sl, tp=tp, confidence=confidence, model=model)

    # B) Consolidado de validación
    bmm = conf.get("best_model_metrics", {})
    metrics_path = bmm.get("file", "")
    sheet = bmm.get("sheet", "metrics")
    criterion = bmm.get("criterion", "Direction_Accuracy")
    mapping = bmm.get("model_to_csv", {})
    model_column_hint = bmm.get("model_column")  # opcional: nombre de columna con el modelo en formato LONG
    metric_name_column_hint = bmm.get("metric_name_column")  # opcional: 'Metric'/'Métrica' en formato WIDE
    wide_format = bmm.get("wide_format")  # opcional: True/False para forzar

    if metrics_path and os.path.exists(metrics_path):
        metrics_df = pd.read_excel(metrics_path, sheet_name=sheet)
        best_model = _pick_best_model_from_metrics(metrics_df, criterion, mapping,
                                                   model_column_hint, metric_name_column_hint, wide_format)
        csv_path = mapping.get(best_model)
        if not csv_path or not os.path.exists(csv_path):
            raise ValueError(f"No hay CSV mapeado para el modelo '{best_model}'. Revisa 'model_to_csv' en el config.")
        last = _read_last_row_csv(csv_path)
        cols = {c.lower(): c for c in last.index}
        side = None
        if "signal" in cols:
            side = str(last[cols["signal"]]).lower()
        elif "direction_pred" in cols:
            dp = float(last[cols["direction_pred"]])
            side = "buy" if dp > 0 else "sell"
        symbol = conf["symbol"]
        entry = float(last[cols["entry"]]) if "entry" in cols and pd.notna(last[cols["entry"]]) else None
        sl    = float(last[cols["sl"]])    if "sl" in cols and pd.notna(last[cols["sl"]]) else None
        tp    = float(last[cols["tp"]])    if "tp" in cols and pd.notna(last[cols["tp"]]) else None
        return Signal(symbol=symbol, side=side, entry=entry, sl=sl, tp=tp, confidence=None, model=best_model, meta={"criterion": criterion})
    raise FileNotFoundError("No se encontró ni 'signal_file' ni 'best_model_metrics.file' válidos en el config.")


# -------------------------------
# Tiempo/sesiones y validaciones
# -------------------------------
def _get_now(conf: dict, mt5):
    tz_str = conf.get("trading_time", {}).get("timezone", None)
    use_server = bool(conf.get("trading_time", {}).get("use_server_time", False))
    symbol = conf.get("symbol")

    tz = ZoneInfo(tz_str) if (tz_str and ZoneInfo) else None

    if use_server and symbol:
        tick = mt5.symbol_info_tick(symbol)
        if tick and getattr(tick, "time", None):
            now_utc = datetime.utcfromtimestamp(tick.time).replace(tzinfo=ZoneInfo("UTC") if ZoneInfo else None)
            return now_utc.astimezone(tz) if tz else now_utc
    now_local = datetime.now(tz) if tz else datetime.now()
    return now_local


def _within_sessions(now_dt: datetime, sessions_cfg: dict) -> bool:
    if not sessions_cfg:
        return True
    wd = now_dt.weekday()  # 0=Mon ... 6=Sun
    hour = now_dt.hour
    curr_names = [k for k, v in DAYMAP.items() if v == wd]
    if not curr_names:
        return True
    day_name = curr_names[0]
    ranges = sessions_cfg.get(day_name, [])
    for st, en in ranges:
        st = int(st); en = int(en)
        if st <= hour < en:
            return True
    return False


def _check_spread_ok(mt5, symbol: str, max_spread_points: float) -> bool:
    sinfo = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not sinfo or not tick:
        return False
    spread = (tick.ask - tick.bid) / sinfo.point
    return spread <= max_spread_points


def _sanitize_volume(mt5, symbol: str, volume: float) -> float:
    sinfo = mt5.symbol_info(symbol)
    if not sinfo:
        return round(max(0.01, volume), 2)
    min_lot  = sinfo.volume_min
    max_lot  = sinfo.volume_max
    step     = sinfo.volume_step
    steps = round(volume / step)
    v = steps * step
    v = max(min_lot, min(v, max_lot))
    return float(f"{v:.2f}")


def build_levels_from_rules(mt5, symbol: str, side: str, rules: dict) -> dict:
    if rules.get("mode", "from_model") == "from_model":
        return {}
    sinfo = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not sinfo or not tick:
        raise RuntimeError("No se pudo leer info/tick del símbolo para calcular niveles.")
    point = sinfo.point

    if rules["mode"] == "fixed_pips":
        sl_pips = float(rules["fixed_pips"]["sl_pips"])
        tp_pips = float(rules["fixed_pips"]["tp_pips"])
        if side == "buy":
            entry = tick.ask
            sl = entry - sl_pips * point
            tp = entry + tp_pips * point
        else:
            entry = tick.bid
            sl = entry + sl_pips * point
            tp = entry - tp_pips * point
        return {"entry": entry, "sl": sl, "tp": tp}
    elif rules["mode"] == "rr":
        sl_pips = float(rules["rr"]["sl_pips"])
        rr_ratio = float(rules["rr"]["rr_ratio"])
        if side == "buy":
            entry = tick.ask
            sl = entry - sl_pips * point
            tp = entry + sl_pips * rr_ratio * point
        else:
            entry = tick.bid
            sl = entry + sl_pips * point
            tp = entry - sl_pips * rr_ratio * point
        return {"entry": entry, "sl": sl, "tp": tp}
    else:
        raise ValueError(f"rules.mode '{rules['mode']}' no soportado.")


# -------------------------------
# Ejecución principal
# -------------------------------
def execute_once(conf: dict) -> Dict[str, Any]:
    # 1) Conexión MT5
    creds = conf["mt5_credentials"]
    mt = Basic_funcs(login=creds["login"], password=creds["password"], server=creds["server"], path=creds.get("path"))

    symbol = conf["symbol"]
    import MetaTrader5 as mt5
    mt5.symbol_select(symbol, True)

    # 2) Validaciones (sesiones y spread)
    now_dt = _get_now(conf, mt5)
    sessions_cfg = conf.get("trading_time", {}).get("sessions", {})
    if not _within_sessions(now_dt, sessions_cfg):
        return {"status": "skipped", "reason": "fuera_de_ventana", "now": str(now_dt)}

    if conf.get("max_spread_points", 0) > 0 and not _check_spread_ok(mt5, symbol, conf["max_spread_points"]):
        return {"status": "skipped", "reason": "spread_excedido", "now": str(now_dt)}

    # 3) Señal
    sig = load_signal_from_files(conf)

    # 4) Niveles
    rules = conf.get("levels_rules", {"mode": "from_model"})
    built = build_levels_from_rules(mt5, sig.symbol, sig.side, rules) if (sig.entry is None or sig.sl is None or sig.tp is None) else {}
    entry = sig.entry if sig.entry is not None else built.get("entry")
    sl    = sig.sl    if sig.sl    is not None else built.get("sl")
    tp    = sig.tp    if sig.tp    is not None else built.get("tp")
    if any(v is None for v in [entry, sl, tp]):
        raise ValueError("No se pudieron determinar entry/SL/TP ni por el archivo ni por las reglas.")

    # 5) Lotaje por riesgo
    risk_pct = float(conf.get("risk_pct", 0.01))
    vol = mt.calculate_position_size(symbol, price_sl=sl, risk_pct=risk_pct)
    vol = _sanitize_volume(mt5, symbol, vol)

    # 6) Ejecutar
    order_type = mt5.ORDER_TYPE_BUY if sig.side == "buy" else mt5.ORDER_TYPE_SELL
    mt.open_operations(par=symbol, volumen=vol, tipo_operacion=order_type, nombre_bot=conf.get("bot_comment", "ExecAgent"),
                       sl=sl, tp=tp)

    result = {
        "status": "sent",
        "symbol": symbol,
        "side": sig.side,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "volume": vol,
        "model": sig.model,
        "confidence": sig.confidence
    }

    # 7) Breakeven
    be_conf = conf.get("breakeven", {})
    if be_conf.get("enabled", False):
        time.sleep(be_conf.get("wait_seconds_after_open", 5))
        n, df_pos = mt.get_opened_positions(par=symbol)
        if n > 0:
            mt.send_to_breakeven(df_pos, perc_rec=float(be_conf.get("progress_pct_to_be", 50.0)))
            result["breakeven_applied"] = True

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta del archivo YAML de configuración")
    ap.add_argument("--run", choices=["once"], default="once", help="Modo de ejecución")
    args = ap.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    result = execute_once(conf)
    print(result)


if __name__ == "__main__":
    main()
