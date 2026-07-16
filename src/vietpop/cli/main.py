# src/vietpop/cli/main.py
import sys

import click
import mlflow
import pandas as pd
from vietpop import __version__
from pathlib import Path
import geopandas as gpd
from libpysal.weights import Queen

from vietpop.core.diagnostics import CommPopDiagnostics, ResidualDiagnostics
from vietpop.utils.config_utils import create_config_template
from vietpop.utils.logger import get_logger
from ..config.settings import Settings
from ..core.feature_extraction import FeatureExtractor
from ..core.merge_communes import CommuneMerger
from ..core.mastergrid import MastergridCreator
from ..core.model import Model
from ..core.dasymetric import DasymetricMapper
from ..utils.raster import remask_layer
from ..core.base_model import \
    EnsembleEstimator, \
    LinearRegressionEstimator, \
    GLMEstimator, \
    RandomForestEstimator, \
    NeuralNetEstimator, \
    BARTEstimator#, PyTorchEstimator
from ..utils.mlflow_tracker import (
    log_diagnostics, setup_mlflow, start_run,
    log_settings, log_estimator_params,
    log_training_metrics, log_model_artifact,
    log_train_artifacts, log_dm_artifacts,
    log_log_file
)

logger = get_logger()

ESTIMATOR_MAP = {
    'rf':      RandomForestEstimator,
    'mlp':     NeuralNetEstimator,
    'lr':      LinearRegressionEstimator,
    'ensemble': EnsembleEstimator,
    'glm':     GLMEstimator,
    'bart':     BARTEstimator,
    # 'pytorch': PyTorchEstimator,
}

@click.group(name='vietpop')
@click.version_option(version=__version__, prog_name='vietpop')
@click.pass_context
def cli(ctx):
    """
    vietpop for geospatial modeling of population distribution.

    A Python toolkit for high-resolution population mapping using machine learning
    and dasymetric techniques.

    For more information, visit: https://vietpop.readthedocs.io/
    """
    ctx.ensure_object(dict)



@cli.command()
@click.option('-c', '--config', 'config_file',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration file')
def mergecommunes(config_file: str):
    """
    Merge communes according to merge_communes settings in config.yaml.
    """
    logger.info(f"Starting commune merging with config: {config_file}")
    settings = Settings.from_file(config_file)
    merger = CommuneMerger(settings)
    outputs = merger.merge()
    logger.info(f"All merged outputs: {outputs}")


@cli.command()
@click.option('-c', '--config', 'config_file',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration file')
@click.option('--force', is_flag=True, help='Force recreate mastergrid even if exists')
def mastergrid(config_file: str, force: bool):
    """
    Create mastergrid raster from admin shapefile.
    """
    logger.info(f"Starting mastergrid creation with config: {config_file}")
    settings = Settings.from_file(config_file)
    creator = MastergridCreator(settings)
    output_path = settings.admin_grid_outfile or (Path(settings.output_dir) / 'mastergrid.tif')
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = Path(settings.output_dir) / output_path
    if force and output_path.exists():
        logger.info(f"Removing existing mastergrid: {output_path}")
        output_path.unlink()
    mastergrid_path = creator.create()
    logger.info(f"Mastergrid created at: {mastergrid_path}")



@cli.command()
@click.option('-c', '--config', 'config_file',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration file')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Show detailed information')
def featurize(config_file: str, verbose: bool) -> None:
    """Extract features only and log to MLflow."""
    logger.info(f"Starting feature extraction workflow with config: {config_file}")
    settings = Settings.from_file(config_file)
    logger.debug(f"Settings: {str(settings)}")

    if verbose:
        logger.set_level('DEBUG')

    output_dir = Path(settings.work_dir) / 'output'
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")

    # MLflow setup
    setup_mlflow(work_dir=str(settings.work_dir))
    with start_run(command='feature_engine',
                   model_type='none',
                   config_file=config_file):
        log_settings(settings)

        feature_extractor = FeatureExtractor(settings)
        logger.info("Extracting features...")
        features = feature_extractor.extract()
        logger.info("Feature extraction completed.")

        # Lưu file features.pkl vào MLflow
        features_pkl = output_dir / 'features.pkl'
        if features_pkl.exists():
            mlflow.log_artifact(str(features_pkl), artifact_path='features')
            logger.info(f"Logged features.pkl to MLflow: {features_pkl}")
        else:
            logger.warning(f"features.pkl not found at {features_pkl}")

        log_log_file(settings.logging['file'] if hasattr(settings, 'logging') and settings.logging else None)

    logger.info("Feature extraction and logging to MLflow completed successfully!")



@cli.command()
@click.option('-c', '--config', 'config_file',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration file')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Show detailed information')
@click.option('--model-type',
              type=click.Choice(list(ESTIMATOR_MAP.keys())),
              default='rf',
              show_default=True,
              help='ML model type to use for training')
def train(config_file: str, 
          verbose: bool, 
          model_type: str) -> None:
    """Train population model and save outputs (no prediction)."""

    logger.info(f"Starting model training workflow with config: {config_file}")
    settings = Settings.from_file(config_file)

    if verbose:
        logger.set_level('DEBUG')

    output_dir = Path(settings.work_dir) / 'output'
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")
     # --- MLflow setup ---
    setup_mlflow(work_dir=str(settings.work_dir))
    estimator_cls = ESTIMATOR_MAP[model_type]
    if model_type == 'glm':
        if not settings.admin:
            msg = "GLM model requires admin_grid.shapefile in config.yaml"
            logger.error(msg)
            raise click.BadParameter(
                msg
            )
        estimator_instance = estimator_cls(shapefile_path=settings.admin)
    else:
        estimator_instance = estimator_cls()

    with start_run(command='train',
                   model_type=model_type,
                   config_file=config_file):

        log_settings(settings)
        log_estimator_params(estimator_instance)

        # Mastergrid check
        if not Path(settings.mastergrid).is_file():
            logger.error(f"Mastergrid not found: {settings.mastergrid}. Please run 'vietpop mastergrid -c config.yaml' first.")
            raise FileNotFoundError(f"Mastergrid not found: {settings.mastergrid}")

        # Remask mastergrid if requested
        if settings.mask:
            logger.info("Remasking mastergrid...")
            outfile = settings.mastergrid.replace('.tif', '_masked.tif')
            remask_layer(settings.mastergrid,
                        settings.mask,
                        1,
                        outfile=outfile,
                        block_size=settings.block_size)
            settings.mastergrid = outfile

        # Constrain mastergrid if requested
        if settings.constrain:
            logger.info("Constraining mastergrid...")
            outfile = settings.mastergrid.replace('.tif', '_constrained.tif')
            remask_layer(settings.mastergrid,
                        settings.constrain,
                        0,
                        outfile=outfile,
                        block_size=settings.block_size)
            settings.constrain = outfile

        feature_extractor = FeatureExtractor(settings)
        model = Model(settings, estimator=estimator_instance, model_type=model_type)
        features = None
        pickle_path = Path(settings.work_dir) / 'output' / 'features.pkl'
        if pickle_path.exists():
            logger.info(f"Loading features from pickle: {pickle_path}")
            features = pd.read_pickle(pickle_path)
        else:
            logger.info("Starting feature extraction...")
            features = feature_extractor.extract()
        if model_type == 'bart':
            logger.info("Training with PyMC-BART estimator")
            model.bart_train(
                data=features,
                model_path=getattr(settings, 'bart_model_path', None),
                scaler_path=getattr(settings, 'bart_scaler_path', None),
                log_scale=settings.log_scale,
                save_model=True,
                draws=getattr(settings, 'draws', 500),
                tune=getattr(settings, 'tune', 1000),
                chains=getattr(settings, 'chains', 2),
                random_seed=getattr(settings, 'random_seed', 42),
            )
        else:
            model.train(features, log_scale=settings.log_scale)

        # Log sau train
        log_training_metrics(model)
        log_model_artifact(
            str(output_dir / f'{model_type}.pkl.gz'),
            str(output_dir / f'{model_type}_scaler.pkl.gz')
        )
        log_train_artifacts(str(output_dir))
        log_log_file(settings.logging['file'] if hasattr(settings, 'logging') and settings.logging else None)

    logger.info("Model training completed successfully!")

@cli.command()
@click.option('-c', '--config', 'config_file',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration file')
@click.option('-m', '--model', 'model_path',
              type=click.Path(exists=True),
              help='Path to model pickle')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Show detailed information')
def predictndm(config_file: str, 
        verbose: bool,
        model_path: str) -> None:
    
    logger.info(f"Starting Prediction and Dasymetric Mapping workflow with config: {config_file}")

    settings = Settings.from_file(config_file)
    logger.debug(f"Settings: {str(settings)}")

    if verbose:
        logger.set_level('DEBUG')

    # Create output directory if it doesn't exist
    output_dir = Path(settings.work_dir) / 'output'
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")

    if not Path(settings.mastergrid).is_file():
        logger.error(f"Mastergrid not found: {settings.mastergrid}. Please run 'vietpop mastergrid -c config.yaml' first.")
        raise FileNotFoundError(f"Mastergrid not found: {settings.mastergrid}")

    # Re-mask mastergrid if requested
    if settings.mask:
        logger.info("Remasking mastergrid...")
        outfile = settings.mastergrid.replace('.tif', '_masked.tif')
        remask_layer(settings.mastergrid,
                     settings.mask,
                     1,
                     outfile=outfile,
                     block_size=settings.block_size)
        settings.mastergrid = outfile

    # Constraining mastergrid if requested
    if settings.constrain:
        logger.info("Constraining mastergrid...")
        outfile = settings.mastergrid.replace('.tif', '_constrained.tif')
        remask_layer(settings.mastergrid,
                     settings.constrain,
                     0,
                     outfile=outfile,
                     block_size=settings.block_size)
        settings.constrain = outfile

    feature_extractor = FeatureExtractor(settings)
    
    if model_path:
        logger.info('Loading pre-trained model')
        features_dummy = feature_extractor.get_dummy()
        model = Model(settings)
        model.train(features_dummy, 
                    model_path=model_path, 
                    scaler_path=model_path.replace('.pkl.gz', '_scaler.pkl.gz'),
                    log_scale=settings.log_scale,
                    save_model=False)
    else:
        raise ValueError("Please provide a pre-trained model with --model option.")

    logger.info("Making predictions...")
    predictions = model.predict_grid(log_scale=settings.log_scale)

    mapper = DasymetricMapper(settings)

    logger.info("Performing dasymetric mapping...")

    if settings.district_dm:
        logger.info("Using district-level dasymetric mapping (map_district)")
        mapper.map_district(predictions)
    else:
        logger.info("Using default dasymetric mapping (map)")
        mapper.map(predictions)

    logger.info("Completed successfully!")

    model_type = Path(model_path).name.replace('.pkl.gz', '')
    setup_mlflow(work_dir=str(settings.work_dir))
    with start_run(command="predictndm", model_type=model_type, config_file=config_file):
        log_settings(settings)
        log_dm_artifacts(str(output_dir))
        log_log_file(settings.logging['file'] if hasattr(settings, 'logging') and settings.logging else None)

    if verbose:
        import traceback
        click.echo(traceback.format_exc(), err=True)
    # raise click.Abort()





@cli.command()
@click.option('-c', '--config', 'config_file',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration file')
@click.option('-m', '--model', 'model_path',
              type=click.Path(exists=True),
              help='Path to model pickle')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Show detailed information')
@click.option('--no-viz',
              is_flag=True,
              help='Skip visualization')
@click.option('--model-type',
              type=click.Choice(list(ESTIMATOR_MAP.keys())),
              default='rf',
              show_default=True,
              help='ML model type to use for training') 
def run(config_file: str, 
        verbose: bool, 
        no_viz: bool,
        model_path: str, 
        model_type: str) -> None:
    """Run the complete population modeling workflow."""
    logger.info(f"Starting population modeling workflow with config: {config_file}")

    settings = Settings.from_file(config_file)
    logger.debug(f"Settings: {str(settings)}")

    if verbose:
        logger.set_level('DEBUG')

    # Create output directory if it doesn't exist
    output_dir = Path(settings.work_dir) / 'output'
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")

    # ── Mastergrid: assume already created ──────────────────
    if not Path(settings.mastergrid).is_file():
        logger.error(f"Mastergrid not found: {settings.mastergrid}. Please run 'vietpop mastergrid -c config.yaml' first.")
        raise FileNotFoundError(f"Mastergrid not found: {settings.mastergrid}")
    # ───────────────────────────────────────────────────────────────────────

    #debugging:
    # sys.exit("Debugging - exiting after mastergrid creation")

    # Re-mask mastergrid if requested
    if settings.mask:
        logger.info("Remasking mastergrid...")
        outfile = settings.mastergrid.replace('.tif', '_masked.tif')
        remask_layer(settings.mastergrid,
                     settings.mask,
                     1,
                     outfile=outfile,
                     block_size=settings.block_size)
        settings.mastergrid = outfile

    # Constraining mastergrid if requested
    if settings.constrain:
        logger.info("Constraining mastergrid...")
        outfile = settings.mastergrid.replace('.tif', '_constrained.tif')
        remask_layer(settings.mastergrid,
                     settings.constrain,
                     0,
                     outfile=outfile,
                     block_size=settings.block_size)
        settings.constrain = outfile

    feature_extractor = FeatureExtractor(settings)
    
    # Run workflow
    if model_path:
        logger.info('Loading pre-trained model')
        features_dummy = feature_extractor.get_dummy()
        model = Model(settings)
        model.train(features_dummy, 
                    model_path=model_path, 
                    scaler_path=model_path.replace('model','scaler'),
                    log_scale=settings.log_scale,
                    save_model=False)
    else:
        logger.info("Starting feature extraction...")
        estimator_cls = ESTIMATOR_MAP[model_type]

        if model_type == 'glm':
            if not settings.admin:
                msg = "GLM model requires admin_grid.shapefile in config.yaml"
                logger.error(msg)
                raise click.BadParameter(
                    msg
                )
            estimator_instance = estimator_cls(shapefile_path=settings.admin)
        else:
            estimator_instance = estimator_cls()

        model = Model(settings, estimator=estimator_instance, model_type=model_type)
        features = None
        pickle_path = Path(settings.work_dir) / 'output' / 'features.pkl'
        if pickle_path.exists():
            logger.info(f"Loading features from pickle: {pickle_path}")
            features = pd.read_pickle(pickle_path)
        else:
            logger.info("Starting feature extraction...")
            features = feature_extractor.extract()
        if model_type == 'bart':
            model.bart_train(features, log_scale=settings.log_scale)
        else:
            model.train(features, log_scale=settings.log_scale)

    logger.info("Making predictions...")
    if model_type == 'bart': 
        predictions = model.predict_bart_grid(log_scale=settings.log_scale)
    else:
        predictions = model.predict_grid(log_scale=settings.log_scale)

    mapper = DasymetricMapper(settings)

    logger.info("Performing dasymetric mapping...")
    if model_type == 'bart': 
        for prediction in predictions.values():
            mapper.map(prediction)
    else:
        mapper.map(predictions)

    if not no_viz:
        logger.info("Creating visualization...")
        from ..utils.visualization import Visualizer
        visualizer = Visualizer(settings)

        viz_paths = {
            'mastergrid': settings.mastergrid,
            'prediction': str(output_dir / 'prediction.tif'),
            'normalized_census': str(output_dir / 'normalized_census.tif'),
            'population': str(output_dir / 'dasymetric.tif')
        }

        for name, path in viz_paths.items():
            if not Path(path).exists():
                error_msg = f"Required file for visualization not found: {name} at {path}"
                logger.error(error_msg)
                raise FileNotFoundError(f"Required file for visualization not found: {name} at {path}")

        # Create visualization
        viz_output = str(output_dir / 'visualization.png')
        visualizer.map_redistribute(
            mastergrid_path=viz_paths['mastergrid'],
            probability_path=viz_paths['prediction'],
            normalize_path=viz_paths['normalized_census'],
            population_path=viz_paths['population'],
            output_path=viz_output,
            vis_params={
                'vmin': [0, 0, 0, 0],
                'vmax': [1300, 250, 1, 250],
                'cmap': 'viridis',
                'titles': ['Zones', 'Probability', 'Normalized Zones', 'Redistributed']
            },
            dpi=300,
            figsize=(15, 5),
            nodata=-99
        )
        logger.info(f"Visualization saved as '{viz_output}'")

    logger.info("Population modeling completed successfully!")

    if verbose:
        import traceback
        click.echo(traceback.format_exc(), err=True)
    raise click.Abort()




@cli.command()
@click.option('-c', '--config', 'config_file',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration file')
@click.option('-m', '--model', 'model_path',
              type=click.Path(exists=True),
              required=True,
              help='Path to pre-trained model pickle')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Show detailed information')
def spatialdiag(config_file: str,
                model_path: str,
                verbose: bool) -> None:
    """Run spatial diagnostics on residuals using a pre-trained model."""

    logger.info(f"Starting spatial diagnostics with config: {config_file}")

    # --- Load settings ---
    settings = Settings.from_file(config_file)
    logger.debug(f"Settings: {str(settings)}")

    if verbose:
        logger.set_level('DEBUG')

    # --- Create output directory ---
    output_dir = Path(settings.work_dir) / 'output'
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    model_type = Path(model_path).stem.replace('.pkl', '')
    setup_mlflow(work_dir=str(settings.work_dir))

    with start_run(command='spatialdiag',
                   model_type=model_type,
                   config_file=config_file,
                   extra_tags={'model_path': str(Path(model_path).name)}):

        log_settings(settings)
        # mlflow.log_param('model_path', model_path)

        # --- Feature extraction ---
        feature_extractor = FeatureExtractor(settings)
        features = None
        pickle_path = Path(settings.work_dir) / 'output' / 'features.pkl'
        if pickle_path.exists():
            logger.info(f"Loading features from pickle: {pickle_path}")
            features = pd.read_pickle(pickle_path)
        else:
            logger.info("features.pkl not found, running feature extraction...")
            features = feature_extractor.extract()

        # --- Load pre-trained model ---
        logger.info(f"Loading pre-trained model from: {model_path}")
        features_dummy = feature_extractor.get_dummy()
        model = Model(settings)
        model.train(features_dummy,
                    model_path=model_path,
                    scaler_path=model_path.replace('.pkl.gz', '_scaler.pkl.gz'),
                    log_scale=settings.log_scale,
                    save_model=False)

        # --- Compute residuals ---
        logger.info("Computing residuals...")
        dens_pred = model.predict_admin(features, log_scale=settings.log_scale)

        shapefile_path = (
            settings.admin_grid['shapefile']
            if hasattr(settings, 'admin_grid') and settings.admin_grid
            else settings.admin
        )
        gdf = gpd.read_file(shapefile_path)

        # Khởi tạo ResidualDiagnostics với dữ liệu gốc
        results = ResidualDiagnostics(
            data=features,
            dens_pred=dens_pred,
            geometry=gdf
        )

        # --- Log diagnostics ---
        moran_global = results.moran_global
        moran_local = results.moran_local

        logger.info(f"Residuals - R2: {results.r2:.4f}, RMSE: {results.rmse:.4f}, MAE: {results.mae:.4f}")
        logger.info(f"Residuals - mean: {results.mean:.4f}, std: {results.std:.4f}")
        logger.info(f"Residuals - skewness: {results.skewness:.4f}, kurtosis: {results.kurtosis:.4f}")
        logger.info(f"Residuals - min: {results.min:.4f}, Q25: {results.q25:.4f}, median: {results.median:.4f}, Q75: {results.q75:.4f}, max: {results.max:.4f}")

        if moran_global:
            logger.info(f"Global Moran's I: {moran_global['I']:.4f}, p-value: {moran_global['p_sim']:.4f}")
            logger.info(f"Global Moran's I - Z(norm): {moran_global['z_norm']:.4f}, Z(rand): {moran_global['z_rand']:.4f}")

        if moran_local:
            logger.info(f"Local Moran's I - mean: {moran_local['mean']:.4f}, std: {moran_local['std']:.4f}")
            logger.info(f"Local Moran's I - skewness: {moran_local['skewness']:.4f}, kurtosis: {moran_local['kurtosis']:.4f}")
            logger.info(f"Local Moran's I - min: {moran_local['min']:.4f}, Q25: {moran_local['q25']:.4f}, median: {moran_local['median']:.4f}, Q75: {moran_local['q75']:.4f}, max: {moran_local['max']:.4f}")
            logger.info(f"Local Moran's I Z - mean: {moran_local['z_mean']:.4f}, std: {moran_local['z_std']:.4f}")
            logger.info(f"Local Moran's I Z - skewness: {moran_local['z_skewness']:.4f}, kurtosis: {moran_local['z_kurtosis']:.4f}")
            logger.info(f"Local Moran's I Z - min: {moran_local['z_min']:.4f}, Q25: {moran_local['z_q25']:.4f}, median: {moran_local['z_median']:.4f}, Q75: {moran_local['z_q75']:.4f}, max: {moran_local['z_max']:.4f}")

        # Log diagnostics
        log_diagnostics(results)
        

    logger.info("Spatial diagnostics completed successfully!")

    if verbose:
        import traceback
        click.echo(traceback.format_exc(), err=True)
    # raise click.Abort()




@cli.command()
@click.option('-c', '--config', 'config_file',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration file')
@click.option('-m', '--model', 'model_path',
              type=click.Path(exists=True),
              help='Path to model pickle')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Show detailed information')
def commpopdiag(config_file: str, model_path: str, verbose: bool) -> None:
    """
    Evaluate commune-level population mapping diagnostics and log results.
    """
    logger.info(f"Starting commune-level diagnostics with config: {config_file}")

    settings = Settings.from_file(config_file)
    if verbose:
        logger.set_level('DEBUG')
    output_dir = Path(settings.work_dir) / 'output'
    output_dir.mkdir(exist_ok=True)

    model_type = Path(model_path).stem.replace('.pkl', '')
    setup_mlflow(work_dir=str(settings.work_dir))
    with start_run(command='commpopdiag',
                   model_type=model_type,
                   config_file=config_file):
        diag = CommPopDiagnostics(settings)
        results = diag.evaluate()
        logger.info("CommPopDiagnostics: Evaluation Results:")
        for k, v in results.items():
            logger.info(f"  {k}: {v}")

        log_settings(settings)
        mlflow.log_metrics({k: float(v) for k, v in results.items() if isinstance(v, (int, float))})
        if 'csv' in results and results['csv']:
            mlflow.log_artifact(results['csv'], artifact_path='csv')
            logger.info(f"CommPopDiagnostics: Output saved to MLflow: {results['csv']}")
        log_log_file(settings.logging['file'] if hasattr(settings, 'logging') and settings.logging else None)
   

    logger.info("Commune-level diagnostics completed successfully!")




@cli.command()
@click.option('-c', '--config', 'config_file',
            type=click.Path(exists=True),
            required=True,
            help='Path to configuration file')
@click.option('-p', '--prediction', 'prediction',
            type=click.Path(exists=True),
            required=True,
            help='Path to prediction layer')
@click.option('-t', '--table', 'table',
            type=click.Path(exists=True),
            required=True,
            help='Path to age-sex structure table')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Show detailed information')

def agesex(config_file: str, 
           prediction: str, 
           table:str, 
           verbose:bool) -> None:
    """Dasymetric redistribution of data with age-sex structure."""

    logger.info(f"Starting age-sex redistribution with config: {config_file}")

    settings = Settings.from_file(config_file)
    logger.debug(f"Settings: {str(settings)}")

    if verbose:
        logger.set_level('DEBUG')

    # Create output directory if it doesn't exist
    output_dir = Path(settings.work_dir) / 'output'
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")

    # Re-mask mastergrid if requested
    if settings.mask:
        logger.info("Remasking mastergrid...")
        outfile = settings.mastergrid.replace('.tif', '_masked.tif')
        settings.mastergrid = outfile
        if not Path(outfile).is_file():
            remask_layer(settings.mastergrid,
                         settings.mask,
                         1,
                         outfile=outfile,
                         block_size=settings.block_size)                

    # Constraining mastergrid if requested
    if settings.constrain:
        logger.info("Constraining mastergrid...")
        outfile = settings.mastergrid.replace('.tif', '_constrained.tif')
        settings.constrain = outfile
        if not Path(outfile).is_file():
            remask_layer(settings.mastergrid,
                        settings.constrain,
                        0,
                        outfile=outfile,
                        block_size=settings.block_size)
            
    mapper = DasymetricMapper(settings)
    mapper.map_agesex(prediction, table)

    logger.info("Age-sex redistribution completed successfully!")

    if verbose:
        import traceback
        click.echo(traceback.format_exc(), err=True)
    raise click.Abort()

@cli.command()
@click.argument('project_dir', type=click.Path())
@click.option('--data-dir', default='data', help='Name of directory containing data files')
@click.option('--prefix', default='test_', help='Prefix for data files')
def init(project_dir: str, data_dir: str, prefix: str):
    """Initialize a new vietpop project with proper structure."""
    try:
        # Create project directory
        project_path = Path(project_dir).resolve()
        project_path.mkdir(parents=True, exist_ok=True)

        # Create directories
        data_path = project_path / data_dir
        data_path.mkdir(exist_ok=True)

        output_path = project_path / 'output'
        output_path.mkdir(exist_ok=True)

        # Create config
        config_path = project_path / "config.yaml"
        create_config_template(
            output_path=config_path,
            data_dir=data_dir,
            prefix=prefix
        )

        logger.info(f"Initialized new vietpop project in {project_dir}")
        logger.info("\nCreated directory structure:")
        logger.info(f"{project_dir}/")
        logger.info("|-- config.yaml")
        logger.info(f"|-- {data_dir}/")
        logger.info("|   |-- (place your input files here)")
        logger.info("|-- output/")

        logger.info("\nExpected input files:")
        logger.info(f"  {prefix}buildingCount.tif")
        logger.info(f"  {prefix}buildingSurface.tif")
        logger.info(f"  {prefix}buildingVolume.tif")
        logger.info(f"  {prefix}mastergrid.tif")
        logger.info(f"  {prefix}admin3.csv")

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise click.Abort()


if __name__ == '__main__':
    cli()



