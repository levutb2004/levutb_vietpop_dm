"""
MLflow tracking utilities for vietpop.
Centralizes all MLflow logic: run naming, logging params/metrics/artifacts.
"""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow

from .logger import get_logger

logger = get_logger()

# Tên experiment mặc định theo từng command
EXPERIMENT_NAMES = {
    'train':       'vietpop_train',
    'run':         'vietpop_run',
    'spatialdiag': 'vietpop_spatialdiag',
}


def setup_mlflow(tracking_uri: Optional[str] = None, work_dir: str = '.') -> None:
    """
    Thiết lập MLflow tracking URI.
    Mặc định lưu vào <work_dir>/mlruns để dễ tìm theo project.
    """
    if tracking_uri is None:
        tracking_uri = f"file:{Path(work_dir).resolve() / 'mlruns'}"
        # db_path = Path(work_dir).resolve() / "mlflow.db"
        # tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")


@contextmanager
def start_run(command: str,
              model_type: str,
              config_file: str,
              extra_tags: Optional[Dict[str, str]] = None):
    """
    Context manager: bắt đầu 1 MLflow run với tên và tags chuẩn.

    Run name format: <command>_<model_type>_<timestamp>
    Ví dụ: train_ensemble_20250101_120000
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{command}_{model_type}_{timestamp}"

    experiment_name = EXPERIMENT_NAMES.get(command, f"vietpop_{command}")
    mlflow.set_experiment(experiment_name)

    tags = {
        'command':    command,
        'model_type': model_type,
        'config':     str(Path(config_file).name),
    }
    if extra_tags:
        tags.update(extra_tags)

    logger.info(f"Starting MLflow run: {run_name} (experiment: {experiment_name})")

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        logger.info(f"MLflow run_id: {run.info.run_id}")
        yield run


def log_settings(settings) -> None:
    """Log tất cả Settings params vào MLflow."""
    params = {
        'work_dir':         str(settings.work_dir),
        'data_dir':         str(settings.data_dir),
        'mastergrid':       str(settings.mastergrid),
        'district_mastergrid': str(settings.district_mastergrid),
        'mask':             str(settings.mask),
        'constrain':        str(settings.constrain),
        'log_scale':        settings.log_scale,
        'output_dir':       str(settings.output_dir),
        'by_block':         settings.by_block,
        'block_size':       str(settings.block_size),
        'max_workers':      settings.max_workers,
        'show_progress':    settings.show_progress,
        'gpu':              settings.gpu,
        'district_dm':      settings.district_dm,
        'admin':            str(settings.admin),
        'admin_column':     settings.admin_column,
        'admin_grid_outfile': str(settings.admin_grid_outfile),
    }

    # Log các dict dạng flatten hoặc serialize
    # Covariate
    params['n_covariates'] = len(settings.covariate)
    params['covariate_names'] = ','.join(settings.covariate.keys())
    for k, v in settings.covariate.items():
        params[f'covariate_{k}'] = v

    # Census
    for k, v in settings.census.items():
        params[f'census_{k}'] = v

    # District census
    for k, v in settings.district_census.items():
        params[f'district_census_{k}'] = v

    # Logging
    for k, v in settings.logging.items():
        params[f'logging_{k}'] = v

    # Output raster
    if settings.output_raster:
        for k, v in settings.output_raster.items():
            params[f'output_raster_{k}'] = v

    # Merge communes
    if settings.merge_communes:
        for k, v in settings.merge_communes.items():
            params[f'merge_communes_{k}'] = v

    mlflow.log_params(params)
    logger.debug("MLflow: settings logged")


def log_estimator_params(estimator) -> None:
    """Log hyperparameters của estimator vào MLflow."""
    estimator_cls = estimator.__class__.__name__
    mlflow.log_param('estimator_class', estimator_cls)

    # Lấy params tùy theo loại estimator
    if hasattr(estimator, '_model'):
        model = estimator._model
        params = {}
        for attr in ['n_estimators', 'max_depth', 'random_state',
                     'max_iter', 'hidden_layer_sizes',
                     'learning_rate', 'n_jobs']:
            val = getattr(model, attr, None)
            if val is not None:
                params[f'model_{attr}'] = str(val)
        mlflow.log_params(params)

    # EnsembleEstimator: log từng sub-model
    elif hasattr(estimator, 'model') and isinstance(estimator.model, dict):
        for name, sub_model in estimator.model.items():
            n_est = getattr(sub_model, 'n_estimators', None)
            if n_est:
                mlflow.log_param(f'{name}_n_estimators', n_est)

    logger.debug(f"MLflow: estimator params logged for {estimator_cls}")


def log_feature_selection(selected_features, importances_df=None) -> None:
    """Log kết quả feature selection."""
    mlflow.log_param('n_selected_features', len(selected_features))
    mlflow.log_param('selected_features', ','.join(str(f) for f in selected_features))

    if importances_df is not None:
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            suffix='_feature_importance.csv',
            delete=False, mode='w'
        ) as f:
            importances_df.to_csv(f, index=False)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path='feature_selection')
        os.unlink(tmp_path)

    logger.debug("MLflow: feature selection logged")


def log_training_metrics(model) -> None:
    """Log target_mean và thông tin train."""
    if hasattr(model, 'target_mean') and model.target_mean is not None:
        mlflow.log_metric('target_mean', float(model.target_mean))
    if hasattr(model, 'feature_names') and model.feature_names is not None:
        mlflow.log_param('n_total_features', len(model.feature_names))
    logger.debug("MLflow: training metrics logged")


def log_cv_scores(scores: Dict[str, float]) -> None:
    """Log cross-validation scores."""
    mlflow.log_metrics({f'cv_{k}': v for k, v in scores.items()})
    logger.debug("MLflow: CV scores logged")


def log_diagnostics(results) -> None:
    """Log SpatialDiagnostics metrics vào MLflow."""
    metrics = {
        'residuals_r2':       results.r2,
        'residuals_rmse':     results.rmse,
        'residuals_mae':      results.mae,
        'residuals_mean':     results.mean,
        'residuals_std':      results.std,
        'residuals_skewness': results.skewness,
        'residuals_kurtosis': results.kurtosis,
        'residuals_min':      results.min,
        'residuals_q25':      results.q25,
        'residuals_median':   results.median,
        'residuals_q75':      results.q75,
        'residuals_max':      results.max,
    }

    moran_global = results.moran_global
    if moran_global:
        metrics.update({
            'moran_global_I':      moran_global['I'],
            'moran_global_p_sim':  moran_global['p_sim'],
            'moran_global_z_norm': moran_global['z_norm'],
            'moran_global_z_rand': moran_global['z_rand'],
        })

    moran_local = results.moran_local
    if moran_local:
        metrics.update({
            'moran_local_mean':     moran_local['mean'],
            'moran_local_std':      moran_local['std'],
            'moran_local_skewness': moran_local['skewness'],
            'moran_local_median':   moran_local['median'],
        })

    mlflow.log_metrics(metrics)
    logger.debug("MLflow: diagnostics metrics logged")


def log_model_artifact(model_path: str, scaler_path: str) -> None:
    """Log model và scaler files như artifacts."""
    if Path(model_path).exists():
        mlflow.log_artifact(model_path, artifact_path='model')
        logger.debug(f"MLflow: model artifact logged: {model_path}")
    if Path(scaler_path).exists():
        mlflow.log_artifact(scaler_path, artifact_path='model')
        logger.debug(f"MLflow: scaler artifact logged: {scaler_path}")

def log_log_file(log_file: str) -> None:
    """Log file log nếu có."""
    if log_file and Path(log_file).exists():
        logger.debug(f"MLflow: log file artifact logged: {log_file}")
        mlflow.log_artifact(str(log_file), artifact_path='logs')

def log_train_artifacts(output_dir: str) -> None:
    output_path = Path(output_dir)
    artifacts = {
        'plots':    ['feature_selection.png'],
        'data':     ['feature_importance.csv', 'features.csv'],
    }
    for artifact_path, files in artifacts.items():
        for fname in files:
            fpath = output_path / fname
            if fpath.exists():
                mlflow.log_artifact(str(fpath), artifact_path=artifact_path)
                logger.debug(f"MLflow: artifact logged: {fpath}")

def log_dm_artifacts(output_dir: str) -> None:
    output_path = Path(output_dir)
    artifacts = {
        'rasters':    ['prediction.tif', 'normalized_census.tif', 'dasymetric.tif'],
    }
    for artifact_path, files in artifacts.items():
        for fname in files:
            fpath = output_path / fname
            if fpath.exists():
                mlflow.log_artifact(str(fpath), artifact_path=artifact_path)
                logger.debug(f"MLflow: artifact logged: {fpath}")
