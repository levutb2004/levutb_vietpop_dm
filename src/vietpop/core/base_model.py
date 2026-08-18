from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count
import numpy as np
import pandas as pd
from typing import Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from quantile_forest import RandomForestQuantileRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pymc as pm
import pymc_bart as pmb
import pickle
from pathlib import Path
# import torch
# import torch.nn as nn
# from skorch import NeuralNetRegressor
# from skorch.callbacks import EarlyStopping, LRScheduler
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import geopandas as gpd
from libpysal.weights import Queen
import gc
import tempfile, os

class BaseEstimator(ABC):
    """
    Abstract base for ML estimator used in population modeling.
    Implement this interface to plug in any ML model.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEstimator':
        """Train the underlying model."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        ...

    @property
    @abstractmethod
    def selected_features(self) -> np.ndarray:
        """Return selected feature names after training."""
        ...

    @selected_features.setter
    @abstractmethod
    def selected_features(self, value: np.ndarray):
        ...

    @property
    def requires_admin_id(self) -> bool:
        """Whether this estimator needs admin IDs for fit/predict."""
        return False

    def set_admin_ids(self, ids: np.ndarray) -> None:
        """Store admin IDs corresponding row-by-row to X/y. No-op by default."""
        pass

class RandomForestEstimator(BaseEstimator):
    """Default estimator: wraps sklearn RandomForestRegressor."""

    def __init__(self, n_estimators: int = 500, random_state: int = 0, **kwargs):
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        self._selected_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestEstimator':
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def selected_features(self):
        return self._selected_features

    @selected_features.setter
    def selected_features(self, value):
        self._selected_features = value
class RandomForestIntervalEstimator(BaseEstimator):

    def __init__(self,n_estimators: int = 500,random_state: int = 0,**kwargs):
        self._model = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        self._selected_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestIntervalEstimator":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict conditional median (q=0.5)."""
        return self._model.predict(X)

    def predict_quantiles(self, X: np.ndarray, quantiles=(0.05, 0.5, 0.95)) -> np.ndarray:
        return self._model.predict(X, quantiles=quantiles)
    @property
    def selected_features(self):
        return self._selected_features

    @selected_features.setter
    def selected_features(self, value):
        self._selected_features = value

class NeuralNetEstimator(BaseEstimator):
    """Example: MLP-based estimator."""

    def __init__(self, hidden_layer_sizes=(20,10,5), max_iter=500, **kwargs):
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            **kwargs
        )
        self._selected_features = None

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    @property
    def selected_features(self):
        return self._selected_features

    @selected_features.setter
    def selected_features(self, value):
        self._selected_features = value

class LinearRegressionEstimator(BaseEstimator):
    """Estimator: wraps sklearn LinearRegression."""

    def __init__(self, **kwargs):
        self._model = LinearRegression(**kwargs)
        self._selected_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionEstimator':
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def selected_features(self):
        return self._selected_features

    @selected_features.setter
    def selected_features(self, value):
        self._selected_features = value

class EnsembleEstimator(BaseEstimator):
    """Ensemble regression using RandomForest, XGBoost, LightGBM."""

    def __init__(self, n_estimators: int = 200, random_state: int = 0):
        self.model = {
            'rf': RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
            'xgb': XGBRegressor(n_estimators=n_estimators, random_state=random_state),
            'lgbm': LGBMRegressor(n_estimators=n_estimators, random_state=random_state)
        }
        self._selected_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleEstimator':
        self.model['rf'].fit(X, y)
        self.model['xgb'].fit(X, y)
        self.model['lgbm'].fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.vstack([
            self.model['rf'].predict(X),
            self.model['xgb'].predict(X),
            self.model['lgbm'].predict(X)
        ])
        return np.mean(preds, axis=0)

    @property
    def selected_features(self):
        return self._selected_features

    @selected_features.setter
    def selected_features(self, value):
        self._selected_features = value


class GLMEstimator(BaseEstimator):
    """
    Geographically Local Model estimator.
    Trains one LinearRegression per admin unit using its 10 spatial neighbours.
    """

    N_NEIGHBOURS = 10

    def __init__(self, shapefile_path: str):
        self._selected_features = None
        self._admin_ids: np.ndarray = None

        self.gdf = gpd.read_file(shapefile_path)
        self.gdf = self.gdf.reset_index(drop=True)
        self.w = Queen.from_dataframe(self.gdf)

        self.gdf['neighbours'] = self.gdf.index.map(
            lambda i: self._find_neighbours(i)
        )

        self.local_models: dict = {
            int(row['OBJECTID']): None
            for _, row in self.gdf.iterrows()
        }

        self._oid_to_idx: dict = {
            int(row['OBJECTID']): idx
            for idx, row in self.gdf.iterrows()
        }

    def _find_neighbours(self, idx: int) -> list:
        """Expand Queen contiguity rings until N_NEIGHBOURS are found."""
        visited: set = set()
        frontier: set = {idx}

        while len(visited) < self.N_NEIGHBOURS + 1:
            next_frontier: set = set()
            for f in frontier:
                for nb in self.w.neighbors.get(f, []):
                    if nb not in visited and nb not in frontier:
                        next_frontier.add(nb)
            visited |= frontier
            if not next_frontier:
                break
            frontier = next_frontier

        visited |= frontier
        visited.discard(idx)
        return list(visited)[:self.N_NEIGHBOURS]

    # admin id bridge

    @property
    def requires_admin_id(self) -> bool:
        return True

    def set_admin_ids(self, ids: np.ndarray) -> None:
        self._admin_ids = ids.astype(int)

    def _resolve_n_jobs(self, n_jobs: int) -> int:
        if n_jobs is None or n_jobs == 1:
            return 1
        if n_jobs == -1:
            return max(1, (cpu_count() or 1) - 1)
        return max(1, int(n_jobs))

    def _build_oid_to_rows(self, duplicate_strategy: str = "last") -> dict:
        if duplicate_strategy not in {"last", "all"}:
            raise ValueError("duplicate_strategy must be one of: {'last', 'all'}")

        if duplicate_strategy == "last":
            # Backward-compatible behavior: duplicate id keeps only last row.
            return {int(oid): [i] for i, oid in enumerate(self._admin_ids)}

        # New behavior: keep all rows for duplicate ids.
        oid_to_rows = defaultdict(list)
        for i, oid in enumerate(self._admin_ids):
            oid_to_rows[int(oid)].append(i)
        return dict(oid_to_rows)

    def _fit_one_local(self, oid: int, xrows: np.ndarray, X: np.ndarray, y: np.ndarray):
        lr = LinearRegression()
        lr.fit(X[xrows], y[xrows])
        return oid, lr, int(xrows.shape[0])

    # BaseEstimator interface

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_jobs: int = 1,
        duplicate_strategy: str = "last",
        include_self_if_no_neighbors: bool = True,
        min_samples: int = 1,
    ) -> 'GLMEstimator':
        if self._admin_ids is None:
            raise RuntimeError(
                "GLMEstimator requires admin ids. "
                "Call set_admin_ids() before fit()."
            )

        if X.shape[0] != len(self._admin_ids) or y.shape[0] != len(self._admin_ids):
            raise ValueError(
                "X/y rows must match number of admin ids. "
                f"Got X={X.shape[0]}, y={y.shape[0]}, ids={len(self._admin_ids)}"
            )

        n_jobs = self._resolve_n_jobs(n_jobs)
        oid_to_rows = self._build_oid_to_rows(duplicate_strategy=duplicate_strategy)

        # Cache neighbors once to reduce repeated geopandas access in loops.
        neighbour_oids_by_oid = {}
        for oid, gdf_idx in self._oid_to_idx.items():
            nb_indices = self.gdf.at[gdf_idx, 'neighbours']
            neighbour_oids_by_oid[oid] = [
                int(self.gdf.at[nb_idx, 'OBJECTID']) for nb_idx in nb_indices
            ]

        # Build training tasks first (deterministic and debuggable).
        tasks = []
        for oid in self._oid_to_idx.keys():
            if oid not in oid_to_rows:
                continue

            xrows = []
            for nb_oid in neighbour_oids_by_oid.get(oid, []):
                rows = oid_to_rows.get(nb_oid)
                if rows:
                    xrows.extend(rows)

            if not xrows and include_self_if_no_neighbors:
                xrows.extend(oid_to_rows[oid])

            if not xrows:
                continue

            # Deduplicate samples when neighbor sets overlap.
            xrows = np.asarray(sorted(set(xrows)), dtype=np.int64)

            if xrows.shape[0] < min_samples:
                continue

            tasks.append((oid, xrows))

        # Reset models for fresh fit.
        for oid in self.local_models.keys():
            self.local_models[oid] = None

        if not tasks:
            return self

        # Sequential path keeps old behavior/perf profile by default.
        if n_jobs == 1 or len(tasks) == 1:
            for oid, xrows in tasks:
                _, lr, _ = self._fit_one_local(oid, xrows, X, y)
                self.local_models[oid] = lr
            return self

        # Thread-based parallelism: low overhead for many small local fits.
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(self._fit_one_local, oid, xrows, X, y)
                for oid, xrows in tasks
            ]
            for fut in as_completed(futures):
                oid, lr, _ = fut.result()
                self.local_models[oid] = lr

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(X.shape[0])

        fitted = [m for m in self.local_models.values() if m is not None]
        if fitted:
            global_coef = np.mean([m.coef_ for m in fitted], axis=0)
            global_intercept = np.mean([m.intercept_ for m in fitted])
        else:
            global_coef = np.zeros(X.shape[1])
            global_intercept = 0.0

        if self._admin_ids is not None and len(self._admin_ids) == X.shape[0]:
            for i, oid in enumerate(self._admin_ids):
                oid = int(oid)
                lm = self.local_models.get(oid)
                if lm is not None:
                    preds[i] = lm.predict(X[i:i + 1])[0]
                else:
                    preds[i] = X[i] @ global_coef + global_intercept
        else:
            preds = X @ global_coef + global_intercept

        return preds

    @property
    def selected_features(self):
        return self._selected_features

    @selected_features.setter
    def selected_features(self, value):
        self._selected_features = value

class BARTEstimator(BaseEstimator):
    """Bayesian Additive Regression Trees (PyMC-BART).

    Việc sampling thực sự được thực hiện trong Model.bart_train()
    thông qua pymc-bart (pmb.BART), KHÔNG thông qua fit()/predict()
    của class này. Class này chỉ giữ vai trò estimator "marker" để
    tương thích với interface chung (ESTIMATOR_MAP, requires_admin_id)
    và lưu trữ metadata liên quan.
    """

    def __init__(self, m: int = 50, random_state: int = 0):
        self.m = m
        self.random_state = random_state
        self._selected_features = None

    def fit(self, X: np.ndarray, y: np.ndarray, 
            draws: int = 250, tune: int = 1000,chains: int = 2, random_seed: int = 42) -> 'BARTEstimator':
        with pm.Model() as bart_model:
            X_data = pm.Data("X_data", X)
            y_data = pm.Data("y_data", y)

            sigma = pm.HalfNormal("sigma", sigma=y.std())
            mu = pmb.BART("mu", X_data, y_data, m=50)
            likelihood = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                return_inferencedata=True,
            )

            self.model = bart_model
            self.trace = idata
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "BARTEstimator.predict() không được dùng trực tiếp. "
            "Việc predict BART được thực hiện thông qua "
            "pm.sample_posterior_predictive() trên trace đã lưu."
        )

    @property
    def selected_features(self):
        return self._selected_features

    @selected_features.setter
    def selected_features(self, value):
        self._selected_features = value
'''
class PopNet(nn.Module):
    """
    PyTorch Neural Network cho population modeling.
    Architecture: FC(100) -> BN -> ReLU -> Dropout
               -> FC(50)  -> BN -> ReLU -> Dropout
               -> FC(25)  -> BN -> ReLU -> Dropout
               -> FC(1)
    """

    def __init__(self, n_features: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(n_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(50, 25),
            nn.BatchNorm1d(25),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output
            nn.Linear(25, 1),
        )

        # Weight initialization (He/Kaiming cho ReLU)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X).squeeze(-1)

class PyTorchEstimator(BaseEstimator):
    """
    Estimator wrapping PopNet (PyTorch) via skorch.
    - Early stopping trên validation loss
    - CUDA tự động nếu có GPU
    - L2 regularization qua weight_decay
    - Compatible hoàn toàn với BaseEstimator interface
    """

    def __init__(
        self,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 15,
        batch_size: int = 128,
        **kwargs
    ):
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.kwargs = kwargs

        self._model = None          # khởi tạo lazy trong fit() khi biết n_features
        self._selected_features = None

    def _build_model(self, n_features: int) -> NeuralNetRegressor:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        callbacks = [
            EarlyStopping(
                monitor='valid_loss',
                patience=self.patience,
                lower_is_better=True
            ),
            LRScheduler(
                policy=ReduceLROnPlateau,
                monitor='valid_loss',
                factor=0.5,
                patience=5,
            ),
        ]

        return NeuralNetRegressor(
            module=PopNet,
            module__n_features=n_features,
            module__dropout=self.dropout,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=self.weight_decay,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            train_split=skorch.helper.predefined_split(  # 10% validation
                skorch.dataset.Dataset(None, None)       # placeholder, xem note bên dưới
            ),
            callbacks=callbacks,
            device=device,
            verbose=1,
            **self.kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PyTorchEstimator':
        # skorch yêu cầu float32
        X_f = X.astype(np.float32)
        y_f = y.astype(np.float32)

        n_features = X_f.shape[1]
        self._model = self._build_skorch(n_features)
        self._model.fit(X_f, y_f)
        return self

    def _build_skorch(self, n_features: int) -> NeuralNetRegressor:
        """Tách riêng để dễ override / test."""
        import skorch
        from skorch.dataset import CVSplit

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        callbacks = [
            EarlyStopping(
                monitor='valid_loss',
                patience=self.patience,
                lower_is_better=True
            ),
            LRScheduler(
                policy=ReduceLROnPlateau,
                monitor='valid_loss',
                factor=0.5,
                patience=5,
            ),
        ]

        return NeuralNetRegressor(
            module=PopNet,
            module__n_features=n_features,
            module__dropout=self.dropout,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=self.weight_decay,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            train_split=CVSplit(cv=0.1, stratified=False),   # 10% validation
            callbacks=callbacks,
            device=device,
            verbose=1,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X.astype(np.float32))

    @property
    def selected_features(self):
        return self._selected_features

    @selected_features.setter
    def selected_features(self, value):
        self._selected_features = value


'''