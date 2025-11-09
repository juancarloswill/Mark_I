# Mark_I — EDA + Modelado + Backtesting (EURUSD / Multimodelo)

> Pipeline para análisis, modelado y backtesting de series de tiempo (FX / intradía y diario) con **ARIMA/SARIMA**, **Prophet** y **LSTM**, controlado via `config.yaml` y con ejecución desde CLI.

## Índice
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Configuración (utils/config.yaml)](#configuración-utilsconfigyaml)
- [Exploración de Datos (EDA)](#exploración-de-datos-eda)
- [Modelos](#modelos)
  - [ARIMA / SARIMA](#arima--sarima)
  - [Prophet](#prophet)
  - [LSTM](#lstm)
  - [Registro de modelos (registry)](#registro-de-modelos-registry)
- [Ejecución del proyecto](#ejecución-del-proyecto)
  - [Modo normal (single-model)](#modo-normal-single-model)
  - [Modo normal MULTIMODELO](#modo-normal-multimodelo)
  - [Modo backtest (single-model)](#modo-backtest-single-model)
  - [Modo backtest MULTIMODELO](#modo-backtest-multimodelo)
- [Resultados y salida](#resultados-y-salida)
- [Utilidades](#utilidades)
- [Validación de configuración](#validación-de-configuración)
- [Solución de problemas](#solución-de-problemas)
- [Licencia](#licencia)

---

## Estructura del proyecto
Ruta y archivos principales (relevantes para este README):

```
Mark_I/
├─ app/
│  ├─ backtesting/
│  │  └─ backtest_rolling.py
│  └─ eda/
│     └─ eda_crispdm.py                (opcional si usas el módulo de EDA)
├─ conexion/
│  └─ easy_Trading.py                  (Basic_funcs – conexión MT5)
├─ modelos/
│  ├─ arima/
│  │  └─ adapter.py                    (entrenar_modelo_arima / predecir_precio_arima)
│  ├─ prophet/
│  │  └─ adapter.py                    (entrenar_modelo_prophet / predecir_precio_prophet)
│  └─ lstm_model.py                    (clase LSTMModel con .fit/.predict)
├─ outputs/
│  ├─ modelos/
│  ├─ backtest_plots/
│  └─ backtest_multi/
├─ scripts/
│  └─ validate_config.py               (validador de config)
├─ utils/
│  ├─ config.yaml                      (configuración principal)
│  └─ __init__.py
├─ registry.py                         (fábrica de modelos / wrappers)
├─ main.py                             (CLI principal)
└─ README.md
```

---

## Requisitos
- **Python 3.10 – 3.11** (Windows 64-bit recomendado para integración con MetaTrader 5).
- Instalar dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> `requirements.txt` incluye: `pandas`, `numpy`, `statsmodels`, `prophet`, `tensorflow-cpu` (para LSTM), `arch` (GARCH opcional), `matplotlib`, `openpyxl/xlsxwriter`, `PyYAML`, `tqdm`, y `MetaTrader5`.

**Nota TF / VS Code**  
Si Pylance no resuelve `tensorflow.keras`, asegúrate de usar el intérprete correcto y, si es necesario, agrega en `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv\Scripts\python.exe",
  "python.analysis.extraPaths": [".venv/Lib/site-packages"],
  "python.analysis.typeCheckingMode": "basic"
}
```

---

## Configuración (`utils/config.yaml`)
El archivo `utils/config.yaml` controla **símbolos**, **modo de trabajo**, **parámetros de modelos** y **backtesting**. Puntos clave:

- **Símbolos / conexión MT5:**
  - `simbolo`, `timeframe`, `cantidad_datos`
  - `mt5: { login, password, server, path }` (usado por `conexion/easy_Trading.Basic_funcs`)

- **EDA:**
  - `eda.frecuencia_resampleo`: `"H"` para intradía, `"D"` para diario.

- **Single-model (compatibilidad):**
  ```yaml
  modelo:
    nombre: "ARIMA"    # "ARIMA" | "SARIMA" | "PROPHET" | "LSTM"
    objetivo: "retornos"  # "retornos" | "nivel"
    horizonte: 5
    params:
      # … parámetros propios del modelo seleccionado …
  ```

- **MULTIMODELO (recomendado):**
  ```yaml
  modelos:
    - name: "PROPHET"
      enabled: true
      objetivo: "retornos"
      horizonte: 5
      params:
        frecuencia_hint: "H"
        interval_width: 0.90
        seasonality_mode: "additive"
        changepoint_prior_scale: 0.05

    - name: "LSTM"
      enabled: true
      objetivo: "retornos"
      horizonte: 5
      params:
        window: 64
        units: 64
        dropout: 0.2
        epochs: 40
        batch_size: 32
        lr: 0.001
        scaler: "standard"
        patience: 5
        model_dir: "outputs/modelos/lstm"
  ```

- **Backtest:**
  ```yaml
  bt:
    initial_train: 1500
    step: 10
    horizon: 1
    target: "returns"          # "returns" | "level"
    pip_size: 0.0001

    threshold_mode: "garch"    # "fixed" | "atr" | "garch"
    threshold_pips: 12.0
    atr_window: 14
    atr_k: 0.60
    garch_k: 0.60
    min_threshold_pips: 10.0
    log_threshold_used: false

    auto:
      scan: {}
      rescan_each_refit: false
      rescan_every_refits: 25

    # Engine se resuelve automáticamente:
    #  - "classic_auto" para ARIMA/SARIMA
    #  - "model" para Prophet/LSTM
  ```

> El `main.py` detecta si usas `modelos:` (multimodelo) o `modelo:` (single). No necesitas editar el YAML entre ejecuciones; basta con activar/desactivar en `modelos.enabled`.

---

## Exploración de Datos (EDA)
Módulo (opcional) en `app/eda/eda_crispdm.py`.  
Se ejecuta con:

```bash
python -m app.main --modo eda --config utils/config.yaml
```

- Resampleo según `eda.frecuencia_resampleo`.
- Gráficos y tablas de diagnóstico (si el módulo está presente).

---

## Modelos

### ARIMA / SARIMA
- **Ruta:** `modelos/arima/adapter.py`
- **Interfaz (funciones):**
  - `entrenar_modelo_arima(df, modo, order, seasonal_order, enforce_stationarity, enforce_invertibility)`
  - `predecir_precio_arima(state, pasos)`
- **Uso en registry:** Wrapper (`registry.py`) convierte la salida (por ejemplo, columnas `timestamp_prediccion` y `precio_estimado`) al formato estándar `yhat` con índice datetime.

### Prophet
- **Ruta:** `modelos/prophet/adapter.py`
- **Interfaz (funciones):**
  - `entrenar_modelo_prophet(df, modo, frecuencia_hint, interval_width, seasonality_mode, changepoint_prior_scale, yearly_seasonality, weekly_seasonality, daily_seasonality)`
  - `predecir_precio_prophet(state, pasos, frecuencia)`
- **Uso en registry:** Wrapper tolerante a nombres de predicción; normaliza a `yhat`.

### LSTM
- **Ruta:** `modelos/lstm_model.py`
- **Interfaz (clase):** `LSTMModel(model_cfg, cfg)` con:
  - `.fit(series: pd.Series)`
  - `.predict(horizon, last_window=None, last_timestamp=None, index=None) -> DataFrame[yhat]`
- Entrena 1-paso y hace *recursive forecasting* para horizontes > 1.
- Scaler ligero configurable (`standard`, `minmax`, `none`).

### Registro de modelos (registry)
- **Ruta:** `registry.py`
- Fábrica `get_model(nombre, cfg)` que devuelve un objeto con interfaz unificada:
  - `.fit(series)` y `.predict(horizon, …) → DataFrame[yhat]`
- Soporta:
  - Adapters por funciones (**ARIMA/SARIMA**, **Prophet**) → envueltos en wrappers.
  - Clase directa (**LSTM**).

---

## Ejecución del proyecto

### Modo normal (single-model)
Usa el bloque `modelo:` del YAML (compatibilidad):

```bash
python -m app.main --modo normal --config utils/config.yaml
```

- Entrena y predice **un** modelo (el definido en `modelo.nombre`).
- Salida: `outputs/modelos/<SIMBOLO>_<MODELO>_forecast.csv`.

### Modo normal MULTIMODELO
Activa varios modelos en `modelos:` (`enabled: true`). Ejecuta:

```bash
python -m app.main --modo normal --config utils/config.yaml
```

- Entrena y predice **todos** los modelos activos (p. ej., Prophet y LSTM en una misma corrida).
- Salidas en `outputs/modelos/`, una por modelo:
  - `EURUSD_PROPHET_forecast.csv`
  - `EURUSD_LSTM_forecast.csv`

**Comparación rápida:** importa esos CSV a tu notebook/Power BI, o usa el mismo índice de fechas para superponer `yhat`.

### Modo backtest (single-model)
Define un modelo en `modelo:` y su configuración en `bt:`:

```bash
python -m app.main --modo backtest --config utils/config.yaml
python -m app.main --modo backtest --config utils/config_optimizado.yaml
```

- Para **ARIMA/SARIMA**, usa el motor **clásico** (`classic_auto`) con umbrales `fixed/atr/garch`.
- Para **Prophet/LSTM**, usa el motor **por clase** (`model`), generando matrices “wide” 1..H.

**Salidas:**
- Clásico (ARIMA/SARIMA):
  - Excel: `outputs/evaluacion.xlsx` (Summary + hojas por modelo)
  - Gráficos: `outputs/backtest_plots/…`
- Por clase (Prophet/LSTM):
  - CSV: `outputs/modelos/<SIMBOLO>_<MODELO>_backtest.csv`

### Modo backtest MULTIMODELO
Activa múltiples modelos en `modelos:` y ejecuta:

```bash
python -m app.main --modo backtest --config utils/config.yaml
```

- **ARIMA/SARIMA:** se evalúan en el motor clásico.
- **Prophet/LSTM:** se evalúan con el motor por clase (**`run_backtest_many`**).
- Salidas en `outputs/backtest_multi/` (una matriz por modelo).

**Interpretación de resultados comparativos:**
- ARIMA/SARIMA (clásico):
  - Revisa `outputs/evaluacion.xlsx` → hoja `Summary`: **RMSE**, **MAE**, **R2**, **HitRate_%**, **Total_pips**, **MaxDD_pips**.
  - Gráficos de **pips vs umbral** y **equity curve** en `outputs/backtest_plots/`.
- Prophet/LSTM (por clase):
  - CSV “wide” donde columnas 1..H son los pasos de predicción; puedes calcular métricas por horizonte y comparar entre modelos en un notebook o Power BI.

---

## Resultados y salida
- **`outputs/modelos/`**
  - `EURUSD_<MODELO>_forecast.csv` (modo normal)
  - `EURUSD_<MODELO>_backtest.csv` (engine=model)
- **`outputs/evaluacion.xlsx`**
  - Resumen y detalle del backtest clásico (ARIMA/SARIMA)
- **`outputs/backtest_plots/`**
  - Gráficos de pips vs umbral y equity
- **`outputs/backtest_multi/`**
  - Matrices wide por modelo (Prophet/LSTM)

---

## Utilidades
- **Conexión MT5:** `conexion/easy_Trading.py` (`Basic_funcs`)
  - `get_data_for_bt(timeframe, symbol, n_barras)` retorna OHLCV para el pipeline.
- **Backtesting:** `app/backtesting/backtest_rolling.py`
  - `evaluate_many` (clásico: ARIMA/SARIMA + umbrales)
  - `run_backtest`, `run_backtest_many` (por clase: Prophet/LSTM)
- **Registry:** `registry.py`
  - Fábrica de modelos unificada (`get_model`).

---

## Validación de configuración
Script: `scripts/validate_config.py`  
Valida la estructura y valores del `utils/config.yaml`.  
Ejemplos:

```bash
# Validación simple
python scripts/validate_config.py utils/config.yaml

# Vista previa con resumen de secciones
python scripts/validate_config.py utils/config.yaml --preview
```

Salida típica (resumida):
```
GENERAL / EDA / MODO NORMAL / BACKTEST …
— ENGINE='model' (PROPHET/LSTM) —
bt.prophet : { … }
bt.lstm    : { … }
✅ Validación completada.
```

---

## Solución de problemas

**Pylance marca `tensorflow.keras` como no encontrado**  
- Asegúrate de que VS Code usa el intérprete correcto (`.venv\Scripts\python.exe`).
- Si persiste, agrega en `.vscode/settings.json`:
  ```json
  {
    "python.defaultInterpreterPath": ".venv\Scripts\python.exe",
    "python.analysis.extraPaths": [".venv/Lib/site-packages"],
    "python.analysis.typeCheckingMode": "basic"
  }
  ```
- Reinicia Pylance: *Pylance: Restart language server* y *Developer: Reload Window*.

**GARCH no disponible**  
- Si `threshold_mode: garch` y no tienes `arch` instalado, cambia a `atr` o `fixed`.

**Intradiario vs Diario**  
- Intradía: `timeframe: "H1"` y `eda.frecuencia_resampleo: "H"`.
- Diario: `timeframe: "D1"` (o resampleo a `"D"` si el feed es intradía).

**MT5**  
- Verifica `mt5.path` a tu terminal y credenciales correctas (`login`, `server`).

---

## Licencia
Este proyecto es de uso académico/experimental. Ajusta esta sección según la licencia que quieras (MIT, Apache-2.0, etc.).
