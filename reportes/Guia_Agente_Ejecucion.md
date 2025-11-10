# Guía del Agente de Ejecución (MT5)

## ¿Qué hace?
1) Lee la **señal** del mejor modelo (o de un CSV/XLSX directo).  
2) Calcula **lotes** por **% riesgo** y **distancia al SL**.  
3) Define **Entry/SL/TP** (usa los del archivo o los construye por reglas).  
4) Valida **ventana horaria** y **spread máximo**.  
5) Envía la **orden a MT5** y puede mover a **Breakeven** automáticamente.

## ¿Cómo decide el lado (buy/sell)?
- Si existe una columna `signal` con `buy`/`sell`, la usa.
- Si existe `direction_pred` (1/-1), lo traduce a `buy`/`sell`.

## Tamaño de la posición
Usa el **balance** de la cuenta, el **% riesgo** y la **distancia al SL** para calcular el **volumen (lotes)**.  
Se ajusta a `volume_min`, `volume_max` y `volume_step` del símbolo.

## Niveles Entry/SL/TP
- Si el archivo trae `entry/sl/tp` → se usan.  
- Si no, se construyen con `levels_rules`:
  - `fixed_pips`: SL/TP por pips fijos.
  - `rr`: SL por pips y TP = SL × `rr_ratio`.

## Reglas de ejecución
- **Ventana horaria** (`trading_window`).  
- **Spread máximo** (`max_spread_points`).  
- **Breakeven** opcional (`breakeven.enabled`).

## Uso
1. Edita `config/execution.yaml` o `config/execution_windows.yaml`.  
2. Ejecuta:
   ```bash
   python -m agents.execution_agent --config config/execution_windows.yaml --run once
   ```

## Integración con tu proyecto
- El agente importa `conexion/easy_Trading.py` (`Basic_funcs`) para conectarse a MT5, enviar y gestionar órdenes.
- Para seleccionar el mejor modelo usa `validacion_consolidado.xlsx` (hoja `metrics`) y mapeos `model_to_csv`.

## Troubleshooting
- **ImportError Basic_funcs**: revisa ruta `conexion/easy_Trading.py` o copia `easy_Trading.py` junto a `agents/execution_agent.py`.  
- **spread_excedido / fuera_de_ventana**: ajusta `max_spread_points` y `trading_window`.  
- **No encuentra archivos**: valida rutas en el YAML.  
- **Volumen muy bajo**: revisa `risk_pct` y distancia SL.
