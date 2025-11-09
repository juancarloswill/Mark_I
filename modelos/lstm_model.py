# modelos/lstm_model.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Alias para evitar imports largos
layers = keras.layers
optimizers = keras.optimizers
callbacks = keras.callbacks


@dataclass
class _LSTMDefaults:
    """Valores por defecto para la configuración del LSTM."""
    window: int = 64
    units: int = 64
    dropout: float = 0.20
    horizon: int = 5
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    loss: str = "mse"          # "mse" | "mae"
    optimizer: str = "adam"    # por ahora usamos Adam
    scaler: str = "standard"   # "standard" | "minmax" | "none"
    patience: int = 5
    model_dir: str = "outputs/modelos/lstm"


def _build_scaler(name: str):
    """
    Devuelve un 'scaler' ligero sin depender de sklearn.
    - standard: (x - mean) / std
    - minmax  : (x - min) / (max - min)
    - none    : identidad
    """
    name = (name or "standard").lower()

    class _Identity:
        def fit(self, x): return self
        def transform(self, x): return x
        def fit_transform(self, x): return x
        def inverse_transform(self, x): return x

    if name == "none":
        return _Identity()

    class _Standard:
        def __init__(self):
            self.mean_ = None
            self.std_ = None
        def fit(self, x):
            x = np.asarray(x, dtype=np.float32)
            self.mean_ = float(np.mean(x))
            self.std_ = float(np.std(x) + 1e-8)
            return self
        def transform(self, x):
            x = np.asarray(x, dtype=np.float32)
            return (x - self.mean_) / self.std_
        def fit_transform(self, x):
            return self.fit(x).transform(x)
        def inverse_transform(self, x):
            x = np.asarray(x, dtype=np.float32)
            return x * self.std_ + self.mean_

    class _MinMax:
        def __init__(self):
            self.min_ = None
            self.max_ = None
        def fit(self, x):
            x = np.asarray(x, dtype=np.float32)
            self.min_ = float(np.min(x))
            self.max_ = float(np.max(x))
            if self.max_ - self.min_ < 1e-8:
                self.max_ = self.min_ + 1e-8
            return self
        def transform(self, x):
            x = np.asarray(x, dtype=np.float32)
            return (x - self.min_) / (self.max_ - self.min_)
        def fit_transform(self, x):
            return self.fit(x).transform(x)
        def inverse_transform(self, x):
            x = np.asarray(x, dtype=np.float32)
            return x * (self.max_ - self.min_) + self.min_

    return _Standard() if name == "standard" else _MinMax()


def _make_supervised(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte una serie 1D en dataset supervisado (X, y) para predicción un paso adelante.

    Parameters
    ----------
    series : np.ndarray
        Serie 1D escalada.
    window : int
        Longitud de ventana temporal.

    Returns
    -------
    X : np.ndarray  [n_samples, window, 1]
    y : np.ndarray  [n_samples]
    """
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window: i])
        y.append(series[i])
    X = np.asarray(X, dtype=np.float32)[..., None]
    y = np.asarray(y, dtype=np.float32)
    return X, y


class LSTMModel:
    """
    LSTM univar para forecasting. Interfaz compatible con el 'registry':

    - fit(series: pd.Series) -> None
    - predict(horizon: int, last_window: Optional[pd.Series], last_timestamp: Optional[pd.Timestamp], index: Optional[pd.DatetimeIndex]) -> pd.DataFrame
    """

    def __init__(self, model_cfg: Dict[str, Any] | None = None, cfg: Dict[str, Any] | None = None) -> None:
        """Inicializa hiperparámetros y rutas de guardado."""
        self.model_cfg = {**_LSTMDefaults().__dict__, **(model_cfg or {})}
        self.cfg = cfg or {}

        # Hiperparámetros
        self.window: int = int(self.model_cfg.get("window"))
        self.units: int = int(self.model_cfg.get("units"))
        self.dropout: float = float(self.model_cfg.get("dropout"))
        self.epochs: int = int(self.model_cfg.get("epochs"))
        self.batch_size: int = int(self.model_cfg.get("batch_size"))
        self.lr: float = float(self.model_cfg.get("lr"))
        self.loss: str = str(self.model_cfg.get("loss"))
        self.optimizer: str = str(self.model_cfg.get("optimizer"))
        self.patience: int = int(self.model_cfg.get("patience"))
        self.model_dir: str = str(self.model_cfg.get("model_dir"))
        self.scaler_name: str = str(self.model_cfg.get("scaler"))

        os.makedirs(self.model_dir, exist_ok=True)

        # Se establecen en fit()
        self._scaler = _build_scaler(self.scaler_name)
        self._model: Optional[keras.Model] = None
        self._train_last_window: Optional[pd.Series] = None
        self._freq: str = str(self.cfg.get("freq", "H"))  # para construir índice futuro

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def fit(self, series: pd.Series) -> None:
        """
        Entrena el LSTM con una serie univariada con índice datetime.

        Parameters
        ----------
        series : pd.Series
            Serie objetivo (orden ascendente).
        """
        s = pd.Series(series).astype(float).dropna()
        if len(s) <= self.window + 1:
            raise ValueError(f"Serie demasiado corta para window={self.window}")

        # Escalado
        s_values = s.values.astype(np.float32)
        s_scaled = self._scaler.fit_transform(s_values)

        # Dataset supervisado
        X, y = _make_supervised(s_scaled, self.window)

        # Arquitectura
        self._model = keras.Sequential([
            layers.Input(shape=(self.window, 1)),
            layers.LSTM(self.units, return_sequences=False),
            layers.Dropout(self.dropout),
            layers.Dense(1)
        ])

        # Optimizador y pérdida
        opt = optimizers.Adam(learning_rate=self.lr) if self.optimizer.lower() == "adam" else optimizers.Adam(learning_rate=self.lr)
        self._model.compile(optimizer=opt, loss=self.loss)

        # ---- Callbacks ----
        es = callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)

        # 👇 Guardamos **solo pesos** para evitar el problema con `options` en formato nativo
        best_weights_path = os.path.join(self.model_dir, "lstm_best.weights.h5")
        ckpt = callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,  # clave para evitar el error
            verbose=0
        )

        # Entrenamiento
        self._model.fit(
            X, y,
            validation_split=0.1,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[es, ckpt]
        )

        # Última ventana (escala original) para predicción recursiva
        self._train_last_window = s.iloc[-self.window:]

        # Cargar los mejores pesos si existen
        try:
            self._model.load_weights(best_weights_path)
        except Exception:
            pass

        # Guardado del modelo completo (formato nativo Keras) **sin** 'options'
        try:
            latest_path = os.path.join(self.model_dir, "lstm_latest.keras")
            self._model.save(latest_path)
        except Exception as e:
            print(f"ℹ️ No se pudo guardar lstm_latest.keras: {e}")

    def predict(
        self,
        horizon: int,
        last_window: Optional[pd.Series] = None,
        last_timestamp: Optional[pd.Timestamp] = None,
        index: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        Predice `horizon` pasos hacia adelante con estrategia recursiva.

        Parameters
        ----------
        horizon : int
            Número de pasos a futuro.
        last_window : Optional[pd.Series]
            Ventana final (longitud `window`) en **escala original**. Si None, usa la del entrenamiento.
        last_timestamp : Optional[pd.Timestamp]
            Último timestamp observado (para construir índice futuro si no viene `index`).
        index : Optional[pd.DatetimeIndex]
            Índice a usar para el DataFrame de salida.

        Returns
        -------
        pd.DataFrame
            DataFrame con columna `yhat` e índice temporal.
        """
        if self._model is None:
            raise RuntimeError("El modelo no está entrenado. Llama primero a .fit().")

        if last_window is None:
            if self._train_last_window is None:
                raise ValueError("Debe proporcionar `last_window` o entrenar primero el modelo.")
            last_window = self._train_last_window

        w = pd.Series(last_window).astype(float).dropna()
        if len(w) != self.window:
            raise ValueError(f"`last_window` debe tener tamaño {self.window}, recibido {len(w)}.")

        # Escalar ventana
        w_scaled = self._scaler.transform(w.values.astype(np.float32))
        seq = w_scaled.reshape(1, self.window, 1)

        preds_scaled = []
        for _ in range(int(horizon)):
            yhat_scaled = float(self._model.predict(seq, verbose=0)[0, 0])
            preds_scaled.append(yhat_scaled)
            # Deslizar ventana: entra la última predicción
            seq = np.roll(seq, shift=-1, axis=1)
            seq[0, -1, 0] = yhat_scaled

        # Volver a escala original
        preds = self._scaler.inverse_transform(np.array(preds_scaled, dtype=np.float32))

        # Índice de salida
        if index is not None:
            idx = pd.DatetimeIndex(index)
        else:
            if last_timestamp is None:
                idx = pd.RangeIndex(start=1, stop=horizon + 1, step=1)
            else:
                try:
                    freq = "H" if str(self._freq).upper().startswith("H") else "D"
                    idx = pd.date_range(start=pd.Timestamp(last_timestamp), periods=horizon + 1, freq=freq)[1:]
                except Exception:
                    idx = pd.RangeIndex(start=1, stop=horizon + 1, step=1)

        return pd.DataFrame({"yhat": preds.reshape(-1)}, index=idx)
    
    def load_pretrained(self) -> bool:
        """
        Carga scaler + modelo desde self.model_dir.
        Retorna True si carga algo útil.
        """
        import json
        loaded = False
        latest = os.path.join(self.model_dir, "lstm_latest.keras")
        best_w = os.path.join(self.model_dir, "lstm_best.weights.h5")
        scalerp = os.path.join(self.model_dir, "scaler.json")

        # scaler
        try:
            if os.path.isfile(scalerp):
                with open(scalerp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                # reconstruir scaler
                name = (d or {}).get("name", "standard")
                if name == "none":
                    self._scaler = _build_scaler("none")
                else:
                    self._scaler = _build_scaler(name)
                    for k, v in (d.items()):
                        if k in ("mean_", "std_", "min_", "max_"):
                            setattr(self._scaler, k, float(v))
                loaded = True
        except Exception as e:
            print(f"ℹ️ No se pudo cargar scaler.json: {e}")

        # modelo
        try:
            if os.path.isfile(latest):
                self._model = keras.models.load_model(latest)
                return True
        except Exception as e:
            print(f"ℹ️ No se pudo cargar lstm_latest.keras: {e}")

        # si no hay modelo completo, intenta pesos
        try:
            if self._model is None:
                self._model = keras.Sequential([
                    layers.Input(shape=(self.window, 1)),
                    layers.LSTM(self.units, return_sequences=False),
                    layers.Dropout(self.dropout),
                    layers.Dense(1)
                ])
                opt = optimizers.Adam(learning_rate=self.lr) if str(self.optimizer).lower()=="adam" else optimizers.Adam(learning_rate=self.lr)
                self._model.compile(optimizer=opt, loss=self.loss)

            if os.path.isfile(best_w):
                self._model.load_weights(best_w)
                loaded = True
        except Exception as e:
            print(f"ℹ️ No se pudieron cargar best weights: {e}")

        return loaded
