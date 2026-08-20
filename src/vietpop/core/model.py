# src/vietpop/core/model.py
import math
import numpy as np
import pandas as pd
import copy
import scipy.ndimage as ndimage
from pymc_bart.utils import _sample_posterior
import matplotlib.pyplot as plt
from pathlib import Path
import gzip, cloudpickle
import joblib
import rasterio
import threading
from typing import Tuple, Optional, Any
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from .base_model import BaseEstimator, GLMEstimator, RandomForestEstimator, QuantileForestIntervalEstimator, BARTEstimator  # thêm import
from rasterio.windows import Window
import pymc as pm
import pymc_bart as pmb
import arviz as az
from ..config.settings import Settings
from ..utils.joblib_manager import joblib_resources
from ..utils.logger import get_logger
from ..utils.matplotlib_utils import with_non_interactive_matplotlib
from ..utils.raster_processing import progress_bar
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from .diagnostics import ResidualDiagnostics
import gc

logger = get_logger()

class Model:
    """
    Population prediction model handler.

    This class manages the training, feature selection, and prediction processes
    for population modeling using Random Forest regression.

    Attributes:
        settings (Settings): Configuration settings for the model
        model (RandomForestRegressor): Trained Random Forest model
        scaler (RobustScaler): Fitted feature scaler
        feature_names (np.ndarray): Names of selected features
        target_mean (float): Mean of target variable for normalization
        output_dir (Path): Directory for saving outputs
    """

    def __init__(self, settings: Settings, estimator: BaseEstimator = None, model_type: str = 'rf'):
        """
        Initialize model handler.

        Args:
            settings: vietpop settings instance
            estimator: ML estimator implementing BaseEstimator.
                       Defaults to RandomForestEstimator.
            model_type: Type of model to use. Defaults to 'rf'.
        """
        self.settings = settings
        self._estimator: BaseEstimator = estimator or RandomForestEstimator()
        self.model = None   # giữ để tương thích với _save_model / load_model
        self.scaler = None
        self.feature_names = None
        self.target_mean = None
        self.model_type = model_type  # <--- thêm thuộc tính này
        self.output_dir = Path(settings.work_dir) / 'output'
        self.output_dir.mkdir(exist_ok=True)

    def train(self,
              data: pd.DataFrame,
              model_path: Optional[str] = None,
              scaler_path: Optional[str] = None,
              log_scale: bool = False,
              save_model: bool = False) -> None:
        """
        Train Random Forest model for population prediction.

        Args:
            data: DataFrame containing features and target variables
                 Must include 'id', 'pop', 'dens' columns
            model_path: Optional path to load pretrained model
            scaler_path: Optional path to load fitted scaler
            log_scale: Whether to train the model with log(dens)
            save_model: Whether to save model after training

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If model loading fails
        """
        data = data.dropna()
        # Pass admin IDs to estimator before dropping id column
        if self._estimator.requires_admin_id and 'id' in data.columns:
            self._estimator.set_admin_ids(data['id'].values)

        drop_cols = np.intersect1d(data.columns.values, ['id', 'pop', 'dens','is_commune'])
        X = data.drop(columns=drop_cols).copy()
        y = data['dens'].values
        if log_scale:
            y = np.log(np.maximum(y, 0.1))
        self.target_mean = y.mean()
        self.feature_names = X.columns.values

        logger.debug(f"Features selected: {self.feature_names.tolist()}")
        logger.debug(f"Target mean: {self.target_mean:.4f}")

        if scaler_path is None:
            logger.info("Creating new scaler")
            self.scaler = RobustScaler()
            self.scaler.fit(X)
        else:
            logger.info(f"Loading scaler from: {scaler_path}")
            with joblib_resources():
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.debug("Scaler loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load scaler: {str(e)}")
                    raise

        if model_path is None:
            logger.info("Training new model")
            X_scaled = self.scaler.transform(X)

            self.model = self._estimator
            logger.debug(f"Initialized {self.model.__class__.__name__}")

            if not isinstance(self.model, (GLMEstimator, QuantileForestIntervalEstimator, BARTEstimator)):
                with joblib_resources():
                    logger.info("Performing feature selection")
                    importances, selected = self._select_features(X_scaled, y)
                    logger.debug(f"Selected {len(selected)} features")
            else:
                selected = X.columns.values
                logger.info("skipping feature selection")
                
            X = X[selected]
            self.selected_features = selected
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            logger.info("Fitting model started")
            self.model.fit(X_scaled, y)
            logger.debug("Model fitting completed")

            with joblib_resources():
                logger.info("Calculating cross-validation scores")
                logger.debug("Skipping CV score calculation in this version")
                # self._calculate_cv_scores(X_scaled, y)

        else:
            logger.info(f"Loading model from: {model_path}")
            with joblib_resources():
                try:
                    model_path_p = Path(model_path)
                    self.model = joblib.load(model_path)
                    self.selected_features = self.model.selected_features
                    logger.debug("Model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load model: {str(e)}")
                    raise

        if save_model:
            logger.info("Saving model and scaler")
            with joblib_resources():
                self._save_model()

        logger.info("Model training completed successfully")
    def bart_train(self,
                    data: pd.DataFrame,
                    model_path: Optional[str] = None,
                    scaler_path: Optional[str] = None,
                    log_scale: bool = False,
                    save_model: bool = True,
                    draws: int = 250,
                    tune: int = 1000,
                    chains: int = 4,
                    random_seed: int = 42) -> None:
        """
        Train PyMC-BART model for population density prediction.

        Args:
            data: DataFrame containing features and target variables.
                Must include 'id', 'pop', 'dens' columns
            model_path: Optional path to load a saved InferenceData (.nc) trace
            scaler_path: Optional path to load fitted scaler
            log_scale: Whether to train the model with log(dens)
            save_model: Whether to save trace/model after training
            draws, tune, chains: MCMC sampling parameters
            random_seed: seed for reproducibility

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If model/trace loading fails
        """
        data = data.dropna()

        if self._estimator.requires_admin_id and 'id' in data.columns:
            self._estimator.set_admin_ids(data['id'].values)

        drop_cols = np.intersect1d(data.columns.values, ['id', 'pop', 'dens', 'is_commune'])
        X = data.drop(columns=drop_cols).copy()
        y = data['dens'].values.astype(float)

        if log_scale:
            y = np.log(np.maximum(y, 0.1))

        self.target_mean = y.mean()
        self.feature_names = X.columns.values
        logger.debug(f"Features selected: {self.feature_names.tolist()}")
        logger.debug(f"Target mean: {self.target_mean:.4f}")

        # --- Scaler (BART is scale-invariant for X, but keep for consistency/pipeline reuse) ---
        if scaler_path is None:
            logger.info("Creating new scaler")
            self.scaler = RobustScaler()
            self.scaler.fit(X)
        else:
            logger.info(f"Loading scaler from: {scaler_path}")
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                logger.error(f"Failed to load scaler: {str(e)}")
                raise

        X_scaled = self.scaler.transform(X)
        self.selected_features = X.columns.values  # BART handles feature relevance internally via variable inclusion

        if model_path is None:
            logger.info("Training new BART model")

            with pm.Model() as bart_model:
                X_data = pm.Data("X_data", X_scaled)
                y_data = pm.Data("y_data", y)

                sigma = pm.HalfNormal("sigma", sigma=y.std())
                mu = pmb.BART("mu", X_data, y_data, m=50)
                likelihood = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

                logger.info("Sampling started")
                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=random_seed,
                    return_inferencedata=True,
                )

            self.model = bart_model
            self.trace = idata
            logger.debug("BART sampling completed")

        else:
            logger.info(f"Loading trace from: {model_path}")
            try:
                self.trace = az.from_netcdf(model_path)
            except Exception as e:
                logger.error(f"Failed to load trace: {str(e)}")
                raise

        if save_model:
            logger.info("Saving trace and scaler")
            self._save_bart_model()

        logger.info("BART training completed successfully")
    def _select_features(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         limit: float = 0.0,
                         plot: bool = True,
                         save: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Select features based on importance using permutation importance.
        """
        logger.debug(f"Selection threshold: {limit}")

        names = self.feature_names
        ymean = self.target_mean

        logger.debug("Fitting initial model for feature importance")
        # Fit wrapper, sau đó lấy sklearn estimator bên trong để dùng với permutation_importance
        self.model.fit(X, y)
        sklearn_estimator = getattr(self.model, '_model', self.model)

        logger.info("Calculating permutation importance")
        result = permutation_importance(
            sklearn_estimator, X, y,
            n_repeats=20,
            n_jobs=2,
            random_state=0,
            scoring='neg_root_mean_squared_error'
        )

        sorted_idx = result.importances_mean.argsort()
        
        importances = pd.DataFrame(
            result.importances[sorted_idx].T / ymean,
            columns=names[sorted_idx],
        )

        selected = importances.columns.values[np.median(importances, axis=0) > limit]

        if plot:
            logger.debug("Creating feature importance plot")
            self._plot_feature_importance(importances, limit)

        if save:
            save_path = Path(self.settings.work_dir) / 'output' / 'feature_importance.csv'
            importances.to_csv(save_path, index=False)
            logger.info(f"Feature importance table saved to: {save_path}")

        logger.info(f"Selected {len(selected)} features out of {len(names)} features")
        logger.info(f"Selected features: {selected}")

        return importances, selected

    @with_non_interactive_matplotlib
    def _plot_feature_importance(self,
                                 importance_df: pd.DataFrame,
                                 limit: float) -> None:
        """
        Create box plot visualization of feature importances.

        Args:
            importance_df: DataFrame with feature importances
            limit: Threshold line to display
        """
        logger.info("Creating feature importance plot")

        sy = importance_df.shape[1] * 0.25 + 0.5
        fig, ax = plt.subplots(1, 1, figsize=(4, sy), dpi=90)

        importance_df.plot.box(
            vert=False,
            whis=5,
            ax=ax,
            color='k',
            sym='.k'
        )

        ax.axvline(x=limit, color='k', linestyle='--', lw=0.5)
        ax.set_xlabel('Decrease in nRMSE')

        plt.tight_layout()
        save_path = Path(self.settings.work_dir) / 'output' / 'feature_selection.png'
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Feature importance plot saved to: {save_path}")

    def _calculate_cv_scores(self,
                             X_scaled: np.ndarray,
                             y: np.ndarray,
                             cv: int = 5) -> None:
        """Calculate and print cross-validation scores."""

        logger.debug(f"CV folds: {cv}")

        scoring = {'r2': (100, 'R2'),
                   'neg_root_mean_squared_error': (-1, 'RMSE'),
                   'neg_mean_absolute_error': (-1, 'MAE')
                   }

        # Lấy sklearn estimator bên trong wrapper để cross_validate có thể clone được
        sklearn_estimator = getattr(self.model, '_model', self.model)

        scores = cross_validate(
            sklearn_estimator, X_scaled, y,
            cv=cv,
            scoring=list(scoring.keys()),
            return_train_score=True,
            n_jobs=1
        )

        for k in ['neg_root_mean_squared_error', 'neg_mean_absolute_error']:
            scoring['n' + k] = (-100, 'n' + scoring[k][1])
            scores['test_n' + k] = scores['test_' + k] / self.target_mean
            scores['train_n' + k] = scores['train_' + k] / self.target_mean

        for k in scoring:
            train = scoring[k][0] * scores[f'train_{k}'].mean()
            test = scoring[k][0] * scores[f'test_{k}'].mean()
            gap = abs(train - test)
            logger.info(f"{k}: {train} and {test}")


    @with_non_interactive_matplotlib
    def predict_grid(self,
                     log_scale: bool = False) -> str:
        """
        Generate predictions using trained model for grid raster.

        Returns:
            str: Path to output prediction raster file

        Raises:
            RuntimeError: If model is not trained
            FileNotFoundError: If input rasters are missing
        """
        logger.info("Starting prediction generation")

        if self.model is None or self.scaler is None:
            logger.error("Model not trained. Call train() first")
            raise RuntimeError("Model not trained. Call train() first.")

        with joblib_resources():
            logger.debug("Opening covariate rasters")
            src = {}
            try:
                for k in self.settings.covariate:
                    src[k] = rasterio.open(self.settings.covariate[k], 'r')
                    logger.debug(f"Opened covariate: {k}")

                # Open mastergrid
                logger.debug("Opening mastergrid")
                mst = rasterio.open(self.settings.mastergrid, 'r')

                # Get profile from mastergrid
                profile = mst.profile.copy()
                profile.update({
                    'dtype': 'float32',
                    'blockxsize': self.settings.block_size[0],
                    'blockysize': self.settings.block_size[1],
                })
                logger.debug("Profile created from mastergrid")

                # Setup locks
                reading_lock = threading.Lock()
                writing_lock = threading.Lock()
                names = self.selected_features
                outfile = Path(self.settings.output_raster['prediction'])
                logger.info(f"Output will be saved to: {outfile}")

                with rasterio.open(outfile, 'w', **profile) as dst:
                    def process(window):
                        df = pd.DataFrame()
                        with reading_lock:
                            for s in src:
                                arr = src[s].read(window=window)[0, :, :]
                                df[s + '_avg'] = arr.flatten()

                        df = df[names]

                        # Make predictions
                        sx = self.scaler.transform(df)
                        yp = self.model.predict(sx)
                        if log_scale:
                            yp = np.exp(yp)
                        res = yp.reshape(arr.shape)

                        with writing_lock:
                            dst.write(res, window=window, indexes=1)

                    if self.settings.by_block:
                        logger.info("Processing by blocks")
                        try:
                            logger.debug("Getting block windows...")
                            block_windows = list(dst.block_windows())
                            logger.debug(f"First block window type: {type(block_windows[0])}")
                            logger.debug(f"First block window content: {block_windows[0]}")

                            windows = []
                            for block_window in block_windows:
                                idx, window = block_window
                                windows.append(window)

                            with ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
                                list(progress_bar(
                                    executor.map(process, windows),
                                    self.settings.show_progress,
                                    len(windows),
                                    desc="Prediction"
                                ))

                        except Exception as e:
                            logger.error(f"Error in block processing setup: {str(e)}")
                            import traceback
                            logger.error(f"Full traceback: {traceback.format_exc()}")
                            raise
                    else:
                        logger.info("Processing entire raster at once")
                        try:
                            full_window = Window(0, 0, dst.width, dst.height)
                            process(full_window)
                        except Exception as e:
                            logger.error(f"Error in full raster processing: {str(e)}")
                            import traceback
                            logger.error(f"Full traceback: {traceback.format_exc()}")
                            raise

            finally:
                logger.debug("Closing mastergrid")
                for k in src:
                    try:
                        src[k].close()
                    except Exception as e:
                        logger.warning(f"Error closing source {k}: {str(e)}")

                if mst is not None:
                    try:
                        mst.close()
                    except Exception as e:
                        logger.warning(f"Error closing mastergrid: {str(e)}")

        logger.info("Prediction completed successfully")
        return str(outfile)
    @with_non_interactive_matplotlib
    def predict_grid_interval(self, log_scale: bool = False) -> dict:
        """Dự đoán Quantile RF theo từng quận/huyện và scale kết quả
        để tổng dân số dự đoán trong mỗi quận/huyện khớp với census.
        """
        logger.info("Starting Quantile RF prediction with district-level dasymetric mapping")

        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call train() first.")
        if not hasattr(self.model, "predict_quantiles"):
            raise RuntimeError(
                "Loaded estimator does not support predict_quantiles(). "
                "Train with Quantile Random Forest."
            )

        quantiles = np.linspace(0.005, 0.995, 100).tolist()
        percentiles = [2.5, 97.5]
        
        id_col = self.settings.census["id_column"]
        pop_col = self.settings.census["pop_column"]
        census_df = pd.read_csv(self.settings.census["path"])
        census = dict(zip(census_df[id_col], census_df[pop_col]))
        del census_df
        gc.collect()

        src, mst, dst_handles = {}, None, {}

        try:
            for k in self.settings.covariate:
                src[k] = rasterio.open(self.settings.covariate[k], "r")

            mst = rasterio.open(self.settings.mastergrid, "r")
            mst_arr = mst.read(1)
            profile = mst.profile.copy()
            profile.update({"dtype": "float32"})

            names = self.selected_features
            base_outfile = Path(self.settings.output_raster["prediction"])
            outfiles = {
                "mean": base_outfile.with_name(base_outfile.stem + "_mean.tif"),
                "p25": base_outfile.with_name(base_outfile.stem + "_p25.tif"),
                "p975": base_outfile.with_name(base_outfile.stem + "_p975.tif"),
            }

            dst_handles = {
                key: rasterio.open(path, "w+", **profile)
                for key, path in outfiles.items()
            }

            logger.info(f"Output will be saved to: {outfiles}")

            nodata = profile.get("nodata", -9999.0)
            init_arr = np.full((mst.height, mst.width), nodata, dtype="float32")
            for handle in dst_handles.values():
                handle.write(init_arr, indexes=1)
            del init_arr

            max_label = int(np.nanmax(mst_arr))
            objects = ndimage.find_objects(
                mst_arr.astype(np.int32), max_label=max_label
            )

            district_ids = [
                d for d in census.keys()
                if 0 < d <= max_label and objects[int(d) - 1] is not None
            ]

            logger.info(
                f"Processing {len(district_ids)} admins "
                f"(of {len(census)} in census table)"
            )

            progress = {"done": 0, "total": len(district_ids)}

            def process_district(district_id):
                row_slice, col_slice = objects[int(district_id) - 1]
                window = Window(
                    col_slice.start,
                    row_slice.start,
                    col_slice.stop - col_slice.start,
                    row_slice.stop - row_slice.start,
                )

                mst_win = mst_arr[row_slice, col_slice]
                district_mask = mst_win == district_id
                out_shape = mst_win.shape

                df = pd.DataFrame({
                    s + "_avg": src[s].read(1, window=window).flatten()
                    for s in src
                })[names]

                valid_mask = df.notna().all(axis=1).values & district_mask.flatten()
                n_pixels = len(df)

                mean_arr = np.full(n_pixels, nodata, dtype="float32")
                pct_arrs = {p: mean_arr.copy() for p in percentiles}

                valid_idx = np.where(valid_mask)[0]

                if valid_idx.size > 0:
                    sx = self.scaler.transform(df.iloc[valid_idx])
                    samples = self.model.predict_quantiles(sx, quantiles=quantiles)
                    samples = samples.T
                    if log_scale:
                        samples = np.exp(samples)

                    census_total = census[district_id]
                    sample_totals = samples.sum(axis=1)

                    factors = np.divide(
                        census_total,
                        sample_totals,
                        out=np.zeros_like(sample_totals, dtype=np.float64),
                        where=sample_totals > 0
                    )
                    
                    samples *= factors[:, None]
                    mean_arr[valid_idx] = samples.mean(axis=0)
                    for p in percentiles:
                        pct_arrs[p][valid_idx] = np.percentile(samples,p,axis=0)

                    del samples, factors

                    existing_mean = dst_handles["mean"].read(1, window=window)
                    new_mean = np.where(district_mask, mean_arr.reshape(out_shape), existing_mean)
                    dst_handles["mean"].write(new_mean, window=window, indexes=1)
                    for p in percentiles:
                        tag = f"p{str(p).replace('.', '')}"
                        existing = dst_handles[tag].read(1, window=window)
                        new_val = np.where(district_mask, pct_arrs[p].reshape(out_shape), existing)
                        dst_handles[tag].write(new_val, window=window, indexes=1)

                progress["done"] += 1
                if progress["done"] % 10 == 0 or progress["done"] == progress["total"]:
                    logger.info(
                        f"Admins progress: {progress['done']}/{progress['total']} "
                        f"({100 * progress['done'] / progress['total']:.1f}%)"
                    )

            logger.info("Processing admins sequentially")

            for district_id in progress_bar(
                district_ids,
                self.settings.show_progress,
                len(district_ids),
                desc="QRF Dasymetric Prediction Interval",
            ):
                process_district(district_id)

        except Exception as e:
            logger.error(f"Error during dasymetric prediction: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

        finally:
            for k in src:
                try:
                    src[k].close()
                except Exception as e:
                    logger.warning(f"Error closing source {k}: {str(e)}")

            if mst is not None:
                try:
                    mst.close()
                except Exception as e:
                    logger.warning(f"Error closing mastergrid: {str(e)}")

            for handle in dst_handles.values():
                try:
                    handle.close()
                except Exception as e:
                    logger.warning(f"Error closing output raster: {str(e)}")

        logger.info("Quantile RF dasymetric prediction completed successfully")

        return {k: str(v) for k, v in outfiles.items()}
    @with_non_interactive_matplotlib
    def predict_bart_grid(self, log_scale: bool = False) -> dict:
        """ Dự đoán BART theo từng quận/huyện (dựa trên mastergrid) và scale
        kết quả để tổng dân số dự đoán trong mỗi quận/huyện khớp với census.

        Returns:
            dict: đường dẫn các raster output (mean, p2.5, p97.5)
        """
        logger.info("Starting BART prediction with district-level dasymetric mapping")

        if self.model is None or self.trace is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call bart_train() first.")

        percentiles = [2.5, 97.5]

        id_col = self.settings.census['id_column']
        pop_col = self.settings.census['pop_column']

        # --- Load census: dict {district_id: population} ---
        census_source = self.settings.census['path']
        census_df = pd.read_csv(census_source)
        census = dict(zip(census_df[id_col], census_df[pop_col]))
        del census_df
        gc.collect()
        
        src = {}
        mst = None
        dst_handles = {}
        thread_local = threading.local()
        def get_thread_model():
                """Mỗi thread cần model riêng vì pymc Model context manager không thread-safe."""
                return copy.deepcopy(self.model)
        try:
            logger.debug("Opening covariate rasters")
            for k in self.settings.covariate:
                src[k] = rasterio.open(self.settings.covariate[k], 'r')

            logger.debug("Opening mastergrid")
            mst = rasterio.open(self.settings.mastergrid, 'r')
            mst_arr = mst.read(1)  # load toàn bộ mastergrid vào RAM (raster 100m/VN ~ vài trăm MB, chấp nhận được)

            profile = mst.profile.copy()
            profile.update({'dtype': 'float32'})

            names = self.selected_features
            base_outfile = Path(self.settings.output_raster['prediction'])
            outfiles = {"mean": base_outfile.with_name(base_outfile.stem + "_mean.tif")}
            for p in percentiles:
                tag = f"p{str(p).replace('.', '')}"
                outfiles[tag] = base_outfile.with_name(f"{base_outfile.stem}_{tag}.tif")

            dst_handles = {key: rasterio.open(path, 'w+', **profile) for key, path in outfiles.items()}
            reading_lock = threading.Lock()
            writing_lock = threading.Lock()
            nodata = profile.get('nodata', -9999.0)

            # --- Ghi nodata cho toàn bộ raster trước, sau đó ghi đè theo từng quận ---
            init_arr = np.full((mst.height, mst.width), nodata, dtype='float32')
            for handle in dst_handles.values():
                handle.write(init_arr, indexes=1)
            del init_arr

            # --- Bounding box mỗi district bằng scipy.ndimage.find_objects (rất nhanh) ---
            max_label = int(np.nanmax(mst_arr))
            objects = ndimage.find_objects(mst_arr.astype(np.int32), max_label=max_label)

            district_ids = [d for d in census.keys()
                            if 0 < d <= max_label and objects[int(d) - 1] is not None]
            logger.info(f"Processing {len(district_ids)} admins (of {len(census)} in census table)")

            progress_lock = threading.Lock()
            progress = {'done': 0, 'total': len(district_ids)}


            def process_district(district_id):
                row_slice, col_slice = objects[int(district_id) - 1]
                window = Window(col_slice.start, row_slice.start,
                                col_slice.stop - col_slice.start,
                                row_slice.stop - row_slice.start)

                mst_win = mst_arr[row_slice, col_slice]
                district_mask = (mst_win == district_id)
                out_shape = mst_win.shape
                with reading_lock:
                    df = pd.DataFrame()
                    for s in src:
                        arr = src[s].read(1, window=window)
                        df[s + '_avg'] = arr.flatten()
                    df = df[names]

                valid_mask = df.notna().all(axis=1).values & district_mask.flatten()
                n_pixels = len(df)
                mean_arr = np.full(n_pixels, nodata, dtype='float32')
                pct_arrs = {p: mean_arr.copy() for p in percentiles}

                valid_idx = np.where(valid_mask)[0]

                if valid_idx.size > 0:
                    X_valid = df.iloc[valid_idx]
                    sx = self.scaler.transform(X_valid)
                    thread_model = get_thread_model()

                    with thread_model:
                        samples = _sample_posterior(
                            all_trees=thread_model.mu.owner.op.all_trees,
                            X=sx,
                            rng=np.random.default_rng(42),
                            size=100
                        )
                    samples = samples.squeeze(-1)
                    if log_scale:
                        samples = np.exp(samples)

                    census_total = census[district_id]
                    sample_totals = samples.sum(axis=1)

                    factors = np.divide(
                        census_total,
                        sample_totals,
                        out=np.zeros_like(sample_totals, dtype=np.float64),
                        where=sample_totals > 0
                    )

                    samples *= factors[:, None]
                    mean_arr[valid_idx] = samples.mean(axis=0)
                    for p in percentiles:
                        pct_arrs[p][valid_idx] = np.percentile(samples,p,axis=0)

                    del samples, factors
                # --- Ghi kết quả: read-modify-write vì bbox các quận có thể chồng nhau ---
                with writing_lock:
                    existing_mean = dst_handles["mean"].read(1, window=window)
                    new_mean = np.where(district_mask, mean_arr.reshape(out_shape), existing_mean)
                    dst_handles["mean"].write(new_mean, window=window, indexes=1)
                    for p in percentiles:
                        tag = f"p{str(p).replace('.', '')}"
                        existing = dst_handles[tag].read(1, window=window)
                        new_val = np.where(district_mask, pct_arrs[p].reshape(out_shape), existing)
                        dst_handles[tag].write(new_val, window=window, indexes=1)

                with progress_lock:
                    progress['done'] += 1
                    if progress['done'] % 10 == 0 or progress['done'] == progress['total']:
                        logger.info(f"Admins progress: {progress['done']}/{progress['total']} "
                                    f"({100*progress['done']/progress['total']:.1f}%)")

            with ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
                list(progress_bar(
                    executor.map(process_district, district_ids),
                    self.settings.show_progress,
                    len(district_ids),
                    desc="BART Dasymetric Prediction"
                ))

        except Exception as e:
            logger.error(f"Error during dasymetric prediction: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        finally:
            for k in src:
                try:
                    src[k].close()
                except Exception as e:
                    logger.warning(f"Error closing source {k}: {str(e)}")
            if mst is not None:
                try:
                    mst.close()
                except Exception as e:
                    logger.warning(f"Error closing mastergrid: {str(e)}")
            for handle in dst_handles.values():
                try:
                    handle.close()
                except Exception as e:
                    logger.warning(f"Error closing output raster: {str(e)}")

        logger.info("BART dasymetric prediction completed successfully")
        return {k: str(v) for k, v in outfiles.items()}

    @with_non_interactive_matplotlib
    def predict_admin(self,
                     data: pd.DataFrame,
                     log_scale: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict and compute residuals for admin units.

        Args:
            data: DataFrame containing features and target variables

        Returns:
            Tuple of (y_true, y_pred, residuals)
        """
        logger.info("Starting admin prediction and residual computation")

        # Drop NA rows
        logger.debug(f"Data shape after dropna: {data.shape}")

        # Pass admin IDs to model before dropping id column
        if self.model.requires_admin_id and 'id' in data.columns:
            self.model.set_admin_ids(data['id'].values)

        # Prepare features and target
        drop_cols = np.intersect1d(data.columns.values, ['id', 'pop', 'dens'])
        X = data.drop(columns=drop_cols).copy()
        y = data['dens'].values
        logger.debug(f"Feature columns: {X.columns.values.tolist()}")

        # Log scale if needed
        if log_scale:
            y = np.log(np.maximum(y, 0.1))
            logger.info("Applied log scaling to target")

        # Use selected features from model
        if self.selected_features is not None:
            X = X[self.selected_features]
            logger.debug(f"Using selected features: {self.selected_features.tolist()}")

        # Scale features
        sx = self.scaler.transform(X)
        logger.info("Features scaled for prediction")

        # Predict
        y_pred = self.model.predict(sx)
        logger.info("Prediction completed")

        # y_true = y
        # residuals = y_true - y_pred
        # logger.info(f"Residuals computed: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

        dens_pred = np.exp(y_pred) if log_scale else y_pred
        logger.info(f"Predicted density: mean={dens_pred.mean():.4f}, std={dens_pred.std():.4f}")


        return dens_pred
    def predict_bart_admin(self,
                            data: pd.DataFrame,
                            model_path: Optional[str] = None,
                            log_scale: bool = False) -> np.ndarray:
        """
        Load a saved BART model (if not already loaded) and predict
        population density for admin units.

        Args:
            data: DataFrame containing features (and optionally id column)
            model_path: Optional path override; if given, forces a reload
            log_scale: Whether the model was trained on log(dens)

        Returns:
            Predicted density array (back-transformed if log_scale=True)
        """
        logger.info("Starting BART admin prediction")

        if self._estimator.requires_admin_id and 'id' in data.columns:
            self._estimator.set_admin_ids(data['id'].values)

        # Prepare features and target
        drop_cols = np.intersect1d(data.columns.values, ['id', 'pop', 'dens', 'is_commune'])
        X = data.drop(columns=drop_cols).copy()
        y = data['dens'].values
        logger.debug(f"Feature columns: {X.columns.values.tolist()}")

        # Log scale if needed
        if log_scale:
            y = np.log(np.maximum(y, 0.1))
            logger.info("Applied log scaling to target")

        # Use selected features from model
        if self.selected_features is not None:
            X = X[self.selected_features]
            logger.debug(f"Using selected features: {self.selected_features.tolist()}")

        # Scale features
        sx = self.scaler.transform(X)
        logger.info("Features scaled for prediction")
        model = self.model
        with model:
            samples = _sample_posterior(
                all_trees=model.mu.owner.op.all_trees,
                X=sx,
                rng=np.random.default_rng(42),
                size=100
            )
        samples = samples.squeeze(-1)
        y_pred = samples.mean(axis=0)
        dens_pred = np.exp(y_pred) if log_scale else y_pred
        logger.info(f"Predicted density: mean={dens_pred.mean():.4f}, std={dens_pred.std():.4f}")

        return dens_pred
    def _save_model(self) -> None:
        """Save model and scaler to disk."""
        scaler_path = self.output_dir / f'{self.model_type}_scaler.pkl.gz'

        try:
            self.model.selected_features = self.selected_features

            if isinstance(self.model, RandomForestIntervalEstimator):
                # rpy2/R objects (rf, train_nodes...) không pickle được qua
                # joblib -> dùng save() riêng của estimator (.rds + .pkl)
                rfpi_base = self.output_dir / self.model_type
                self.model.save(str(rfpi_base))
                logger.debug(f"RF-PI model saved to: {rfpi_base}.rds / {rfpi_base}.pkl")
            else:
                model_path = self.output_dir / f'{self.model_type}.pkl.gz'
                joblib.dump(self.model, model_path)
                logger.debug(f"Model saved to: {model_path}")

            joblib.dump(self.scaler, scaler_path)
            logger.debug(f"Scaler saved to: {scaler_path}")

            logger.info("Model and scaler saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model or scaler: {str(e)}")
            raise
    def _save_bart_model(self) -> None:
        """Save BART model (incl. trace/trees), scaler, and metadata to disk in one file."""
        model_path = self.output_dir / f'{self.model_type}.pkl.gz'


        try:
            self.model.feature_names = self.feature_names
            self.model.selected_features = self.selected_features

            with gzip.open(model_path, 'wb') as f:
                cloudpickle.dump({
                    'model': self.model,
                    'trace': self.trace,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'selected_features': self.selected_features,
                    'target_mean': self.target_mean,
                }, f)

            logger.debug(f"Model saved to: {model_path}")
            logger.info("BART model, trace and scaler saved successfully")
        except Exception as e:
            logger.error(f"Failed to save BART model/trace/scaler: {str(e)}")
            raise
    def load_model(self,
                   model_path: str = None,
                   scaler_path: str = None) -> None:
        """
        Load saved model and scaler.

        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        logger.info("Loading saved model and scaler")

        try:
            if model_path is None:
                model_path = self.output_dir / f'{self.model_type}.pkl.gz'
            if scaler_path is None:
                scaler_path = self.output_dir / f'{self.model_type}_scaler.pkl.gz'

            logger.debug(f"Loading model from: {model_path}")
            model_path_p = Path(model_path)
            is_rfpi_file = (model_path_p.suffix == '.pkl'
                             and not model_path_p.name.endswith('.pkl.gz'))
            if isinstance(self._estimator, RandomForestIntervalEstimator) or is_rfpi_file:
                rfpi_base = str(model_path_p.with_suffix(''))
                self.model = RandomForestIntervalEstimator.load(rfpi_base)
            else:
                self.model = joblib.load(model_path)
            self.selected_features = self.model.selected_features
            logger.debug(f"Selected features loaded from model: {self.selected_features}")

            logger.debug(f"Loading scaler from: {scaler_path}")
            self.scaler = joblib.load(scaler_path)

            self.feature_names = self.scaler.get_feature_names_out()
            logger.debug(f"Loaded feature names: {self.feature_names.tolist()}")

            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model or scaler: {str(e)}")
            raise
    def _load_bart_model(self, model_path: Optional[str] = None) -> None:
        try:
            if model_path is None:
                model_path = self.output_dir / f'{self.model_type}.pkl.gz'
            with gzip.open(model_path, 'rb') as f:
                obj = cloudpickle.load(f)
            self.model = obj['model']
            self.trace = obj['trace']
            self.scaler = obj['scaler']
            self.feature_names = obj['feature_names']
            self.selected_features = obj['selected_features']
            self.target_mean = obj['target_mean']
            logger.info(f"BART model, trace and scaler loaded from: {path}")
        except Exception as e:
            logger.error(f"Failed to load BART model/trace/scaler: {str(e)}")
            raise
    def compute_residuals(self,
                          data: pd.DataFrame,
                          log_scale: bool = False) -> ResidualDiagnostics:
        """
        Compute residuals for the current model on input data.

        Args:
            data: DataFrame containing features and target variables
        """
        logger.info("Starting residual computation")

        # Drop NA rows
        data = data.dropna()
        logger.debug(f"Data shape after dropna: {data.shape}")

        # Pass admin IDs to model before dropping id column
        if self.model.requires_admin_id and 'id' in data.columns:
            self.model.set_admin_ids(data['id'].values)

        # Prepare features and target
        drop_cols = np.intersect1d(data.columns.values, ['id', 'pop', 'dens'])
        X = data.drop(columns=drop_cols).copy()
        y = data['dens'].values
        logger.debug(f"Feature columns: {X.columns.values.tolist()}")

        # Log scale if needed
        if log_scale:
            y = np.log(np.maximum(y, 0.1))
            logger.info("Applied log scaling to target")

        # Use selected features from model
        if self.selected_features is not None:
            X = X[self.selected_features]
            logger.debug(f"Using selected features: {self.selected_features.tolist()}")

        # Scale features
        sx = self.scaler.transform(X)
        logger.info("Features scaled for prediction")

        # Predict
        y_pred = self.model.predict(sx)
        logger.info("Prediction completed")


        y_true = y
        # If log_scale, revert prediction to original scale for residuals
        # if log_scale:
        #     y_pred = np.exp(y_pred)
        #     y_true = np.exp(y)
        #     logger.info("Reverted log scale for residuals")

        # Compute residuals
        residuals = y_true - y_pred
        logger.info(f"Residuals computed: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

        # Add residuals to DataFrame
        # data = data.copy()
        # data['y_true'] = y_true
        # data['y_pred'] = y_pred
        # data['residuals'] = residuals

        logger.info(f"Residuals computed: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

        return ResidualDiagnostics(
            data=data,
            y_true=y_true,
            y_pred=y_pred,
            residuals=residuals,
            log_scale=log_scale
        )