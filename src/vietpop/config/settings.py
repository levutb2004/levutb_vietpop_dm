# src/vietpop/config/settings.py
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import math
import pandas as pd
import rasterio
import yaml
from datetime import datetime

from vietpop.utils.logger import get_logger

logger = get_logger()

class Settings:
    """
    Configuration settings manager for vietpop.

    This class handles all configuration settings for population modeling,
    including file paths, processing parameters, and validation of inputs.

    Attributes:
        work_dir (Path): Working directory path
        data_dir (Path): Data directory path
        mastergrid (str): Path to mastergrid file or 'create'
        mask (str): Path to (water) mask file
        constrain (str): Path to raster for constraining population distribution
        covariate (dict): Dictionary of covariate names and paths
        census (dict): Census configuration including path and column names
        log_scale (bool): Whether to train model on log(dens)
        output_dir (Path): Output directory path
        by_block (bool): Whether to process by blocks
        block_size (tuple): Size of processing blocks (width, height)
        max_workers (int): Maximum number of parallel workers
        show_progress (bool): Whether to show progress bars
        gpu (bool): Whether to use GPU acceleration
        admin (str): Path to admin shapefile (for mastergrid creation)
        admin_column (str): Name of admin column in the admin shapefile
        merge_communes (dict): Dictionary of communes to merge
        output_raster (dict): Dictionary of output rasters and paths

    Raises:
        ValueError: If required settings are missing or invalid
        FileNotFoundError: If required files don't exist
    """
    def __init__(self,
                 work_dir: str = ".",
                 data_dir: str = "data",
                 mastergrid: Optional[str] = None,
                 district_mastergrid: Optional[str] = None,
                 mask: Optional[str] = None,
                 constrain: Optional[str] = None,
                 covariates: Optional[Dict[str, str]] = None,
                 census_data: Optional[str] = None,
                 census_pop_column: Optional[str] = None,
                 census_id_column: Optional[str] = None,
                 district_census_data: Optional[str] = None,
                 district_census_pop_column: Optional[str] = None,
                 district_census_id_column: Optional[str] = None,
                 log_scale: bool = True,
                 output_dir: Optional[str] = None,
                 by_block: bool = True,
                 block_size: Tuple[int, int] = (2048, 2048),
                 max_workers: int = 8,
                 show_progress: bool = True,
                 logging: Optional[Dict] = None,
                 gpu: bool = False,
                 district_dm: bool = False,
                 admin_grid: Optional[Dict] = None,
                 merge_communes: Optional[Dict] = None,
                 output_raster: Optional[Dict[str, str]] = None):   

        """
        Initialize Settings with configuration parameters.

        Args:
            work_dir: Root directory for the project
            data_dir: Directory containing input data files
            mastergrid: Path to mastergrid file or 'create'
            district_mastergrid: Path to district mastergrid file or 'create'
            mask (str): Path to (water) mask file
            constrain (str): Path to raster for constraining population distribution
            covariates: Dictionary mapping covariate names to file paths
            census_data: Path to census data file
            census_pop_column: Name of population column in census data
            census_id_column: Name of ID column in census data
            district_census_data: Path to district census data file
            district_census_pop_column: Name of population column in district census data
            district_census_id_column: Name of ID column in district census data
            log_scale (bool): Whether to train model on log(dens)
            output_dir: Directory for output files
            by_block: Whether to process by blocks
            block_size: Tuple of (width, height) for processing blocks
            max_workers: Number of parallel processing workers
            show_progress: Whether to display progress bars
            gpu: Whether to use GPU acceleration
            district_dm: Whether to use district mastergrid
            admin_grid: Path to admin shapefile (for mastergrid creation)
            merge_communes: Dictionary of communes to merge
            output_raster: Dictionary of output rasters and paths

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        logger.info("Initializing vietpop settings")

        # Convert working directory to absolute path
        self.work_dir = Path(work_dir).resolve()
        self.data_dir = self.work_dir / data_dir

        # Handle mastergrid path
        self.mastergrid = str(Path(mastergrid)) if mastergrid else None
        if self.mastergrid and self.mastergrid != 'create':
            if not Path(self.mastergrid).is_absolute():
                self.mastergrid = str(self.data_dir / mastergrid)

        # Handle district mastergrid path
        self.district_mastergrid = str(Path(district_mastergrid)) if district_mastergrid else None
        if self.district_mastergrid and self.district_mastergrid != 'create':
            if not Path(self.district_mastergrid).is_absolute():
                self.district_mastergrid = str(self.data_dir / district_mastergrid)

        # Handle (water) mask path
        self.mask = str(Path(mask)) if mask else None
        if self.mask:
            if not Path(self.mask).is_absolute():
                self.mask = str(self.data_dir / mask)

        # Handle constrain path
        self.constrain = str(Path(constrain)) if constrain else None
        if self.constrain:
            if not Path(self.constrain).is_absolute():
                self.constrain = str(self.data_dir / constrain)

        # Process covariate paths
        self.covariate = {}
        if covariates:
            for key, path in covariates.items():
                if not Path(path).is_absolute():
                    path = str(self.data_dir / path)
                self.covariate[key] = path

        if not self.covariate:
            raise ValueError("At least one covariate is required")

        # Process census path
        census_path = Path(census_data) if census_data else None
        self.census = {
            'path': str(self.data_dir / census_data) if census_data and not census_path.is_absolute() else census_data,
            'pop_column': census_pop_column,
            'id_column': census_id_column
        }

        # Process district census path
        district_census_path = Path(district_census_data) if district_census_data else None
        self.district_census = {
            'path': str(self.data_dir / district_census_data) if district_census_data and not district_census_path.is_absolute() else district_census_data,
            'pop_column': district_census_pop_column,
            'id_column': district_census_id_column
        }

        # Set log scale
        self.log_scale = log_scale

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            if not self.output_dir.is_absolute():
                self.output_dir = self.work_dir / output_dir
        else:
            self.output_dir = self.work_dir / 'output'

        # Set processing parameters
        self.by_block = by_block
        self.block_size = tuple(block_size)
        self.max_workers = max_workers
        self.show_progress = show_progress
        self.gpu = gpu
        self.district_dm = district_dm

        self.logging = {
            'level': 'INFO',
            'file': 'vietpop.log'
        }
        if logging:
            self.logging.update(logging)

        if self.logging['file']:
            # append timestamp to logfile name to avoid overwriting previous runs
            orig = Path(self.logging['file'])
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            logfile = f"{orig.stem}_{ts}{orig.suffix}"
            self.logging['file'] = str(self.output_dir / logfile)
            logger.set_log_file(self.logging['file'])
        logger.set_level(self.logging['level'])

        # Handle admin grid configuration (for mastergrid creation)
        self.admin = None
        self.admin_column = 'OBJECTID'  # default
        self.admin_grid_outfile = None
        if admin_grid and isinstance(admin_grid, dict):
            shapefile = admin_grid.get('shapefile')
            if shapefile:
                admin_path = Path(shapefile)
                if not admin_path.is_absolute():
                    admin_path = self.work_dir / shapefile
                self.admin = str(admin_path)
            self.admin_column = admin_grid.get('column', 'OBJECTID')
            self.admin_grid_outfile = admin_grid.get('outfile')

        # Handle merge_communes configuration
        self.merge_communes = None
        if merge_communes and isinstance(merge_communes, dict):
            shapefile = merge_communes.get('shapefile')
            if shapefile:
                shp_path = Path(shapefile)
                if not shp_path.is_absolute():
                    shp_path = self.data_dir / shapefile
                merge_communes['shapefile'] = str(shp_path)
            outfile = merge_communes.get('outfile')
            if outfile:
                out_path = Path(outfile)
                if not out_path.is_absolute():
                    merge_communes['outfile'] = str(self.output_dir / outfile)
            csv_outfile = merge_communes.get('csv_outfile')
            if csv_outfile:
                csv_path = Path(csv_outfile)
                if not csv_path.is_absolute():
                    merge_communes['csv_outfile'] = str(self.output_dir / csv_outfile)
            self.merge_communes = merge_communes

        # Handle output_raster
        self.output_raster = {}
        if output_raster and isinstance(output_raster, dict):
            for key, val in output_raster.items():
                # Nếu là đường dẫn tương đối, lưu vào output_dir
                if not Path(val).is_absolute():
                    self.output_raster[key] = str(self.output_dir / val)
                else:
                    self.output_raster[key] = val

        # Validate all settings
        self._validate_settings()
        logger.info("Settings initialization completed")

    def _validate_settings(self) -> None:
        """
        Validate settings and check file existence.

        Performs comprehensive validation of all settings including:
        - Required paths and parameters
        - File existence
        - Raster compatibility (CRS, resolution, dimensions)
        - Census data format and required columns

        Raises:
            ValueError: If settings are invalid
            FileNotFoundError: If required files don't exist
        """

        logger.info("Validating settings...")

        if not self.census['path']:
            logger.error("Census data path is required")
            raise ValueError("Census data path is required")
        if not self.census['pop_column']:
            logger.error("Census population column name is required")
            raise ValueError("Census population column name is required")
        if not self.census['id_column']:
            logger.error("Census ID column name is required")
            raise ValueError("Census ID column name is required")
        if not self.covariate:
            logger.error("At least one covariate is required")
            raise ValueError("At least one covariate is required")

        template_profile = None

        if self.mastergrid != 'create':
            if not Path(self.mastergrid).is_file():
                logger.error(f"Mastergrid file not found: {self.mastergrid}")
                raise FileNotFoundError(f"Mastergrid file not found: {self.mastergrid}")
            with rasterio.open(self.mastergrid) as src:
                template_profile = src.profile
                logger.debug("Mastergrid template profile loaded")

        if self.district_mastergrid is not None and self.district_mastergrid != 'create':
            if not Path(self.district_mastergrid).is_file():
                logger.error(f"District mastergrid file not found: {self.district_mastergrid}")
                raise FileNotFoundError(f"District mastergrid file not found: {self.district_mastergrid}")

        if self.mask is not None:
            mask_path = Path(self.mask)
            if not mask_path.is_file():
                logger.error(f"Mask file not found: {self.mask}")
                raise FileNotFoundError(f"Mask file not found: {self.mask}")

        if self.constrain is not None:
            constrain_path = Path(self.constrain)
            if not constrain_path.is_file():
                logger.warning(f"Constraining file not found: {self.constrain}, proceeding without constrain")
                self.constrain = None

        logger.info("Validating covariates...")
        for name, path in self.covariate.items():
            if not Path(path).is_file():
                logger.error(f"Covariate file not found: {path} ({name})")
                raise FileNotFoundError(f"Covariate file not found: {path} ({name})")

            with rasterio.open(path) as src:
                if template_profile is None:
                    template_profile = src.profile
                else:
                    if src.crs != template_profile['crs']:
                        logger.warning(f"Covariate {name}: CRS mismatch")
                    if not(math.isclose(src.transform[0], template_profile['transform'][0], rel_tol=1e-9)):
                        logger.warning(f"Covariate {name}: Resolution mismatch")
                    if src.width != template_profile['width']:
                        logger.warning(f"Covariate {name}: Width mismatch")
                    if src.height != template_profile['height']:
                        logger.warning(f"Covariate {name}: Height mismatch")

        logger.info("Validating census data...")
        census_path = Path(self.census['path'])
        if not census_path.is_file():
            logger.error(f"Census file not found: {census_path}")
            raise FileNotFoundError(f"Census file not found: {census_path}")

        if census_path.suffix.lower() != '.csv':
            logger.error("Census file must be CSV format")
            raise ValueError("Census file must be CSV format")

        try:
            df = pd.read_csv(census_path, nrows=1)
            missing_cols = []
            for col in [self.census['pop_column'], self.census['id_column']]:
                if col not in df.columns:
                    missing_cols.append(col)
                if missing_cols:
                    logger.error(f"Missing required columns in census data: {', '.join(missing_cols)}")
                    raise ValueError(f"Missing required columns in census data: {', '.join(missing_cols)}")
        except Exception as e:
            logger.error(f"Error reading census file: {str(e)}")
            raise ValueError(f"Error reading census file: {str(e)}")

        logger.info("Validating district census data...")
        if self.district_census['path']:
            district_census_path = Path(self.district_census['path'])
            if not district_census_path.is_file():
                logger.error(f"District census file not found: {district_census_path}")
                raise FileNotFoundError(f"District census file not found: {district_census_path}")

            if district_census_path.suffix.lower() != '.csv':
                logger.error("District census file must be CSV format")
                raise ValueError("District census file must be CSV format")

            try:
                df = pd.read_csv(district_census_path, nrows=1)
                missing_cols = []
                for col in [self.district_census['pop_column'], self.district_census['id_column']]:
                    if col not in df.columns:
                        missing_cols.append(col)
                    if missing_cols:
                        logger.error(f"Missing required columns in district census data: {', '.join(missing_cols)}")
                        raise ValueError(f"Missing required columns in district census data: {', '.join(missing_cols)}")
            except Exception as e:
                logger.error(f"Error reading district census file: {str(e)}")
                raise ValueError(f"Error reading district census file: {str(e)}")

        logger.info("Settings validation completed successfully")


    @classmethod
    def validate_config_file(cls, config_path: str) -> None:
        """
        Validate configuration file structure.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            ValueError: If configuration file is missing required fields
                       or has invalid structure
        """
        logger.info(f"Validating configuration file: {config_path}")

        required_fields = {
            'work_dir', 'covariates', 'census_data',
            'census_pop_column', 'census_id_column'
        }

        with open(config_path) as f:
            config = yaml.safe_load(f)

        missing = required_fields - set(config.keys())
        if missing:
            logger.error(f"Missing required fields in config: {missing}")
            raise ValueError(f"Missing required fields in config: {missing}")

        if not isinstance(config.get('covariates', {}), dict):
            logger.error("'covariates' must be a dictionary")
            raise ValueError("'covariates' must be a dictionary")

        logger.info("Configuration file validation successful")

    @classmethod
    def from_file(cls, config_path: str) -> 'Settings':
        """
        Create Settings instance from configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Settings: Initialized Settings instance

        Raises:
            ValueError: If configuration file is invalid
        """
        logger.info(f"Loading settings from file: {config_path}")

        cls.validate_config_file(config_path)

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Resolve work_dir relative to config file location
        config_dir = Path(config_path).parent.resolve()
        if config['work_dir'] == '.':
            config['work_dir'] = str(config_dir)
        elif not Path(config['work_dir']).is_absolute():
            config['work_dir'] = str(config_dir / config['work_dir'])

        # Ensure gpu flag is present and boolean
        config['gpu'] = bool(config.get('gpu', False))
        # Ensure district_dm flag is present and boolean
        config['district_dm'] = bool(config.get('district_dm', False))

        # admin_grid is passed as-is (dict), no transformation needed
        # Remove old flat keys if someone passes them directly
        config.pop('admin', None)
        config.pop('admin_column', None)

        # Đảm bảo output_raster là dict hoặc None
        config['output_raster'] = config.get('output_raster', None)

        logger.info("Settings loaded successfully")
        return cls(**config)

    def __str__(self) -> str:
        """
        Create string representation of settings.

        Returns:
            str: Formatted string containing all settings
        """

        covariate_str = '\n    '.join(f"- {key}: {value}" for key, value in self.covariate.items())
        admin_grid_str = f"    Shapefile: {self.admin}\n    Column: {self.admin_column}" if self.admin else "    None"
        output_raster_str = '\n    '.join(f"- {k}: {v}" for k, v in self.output_raster.items())
        return (
            f"vietpop Settings:\n"
            f"  Work Directory: {self.work_dir}\n"
            f"  Output Directory: {self.output_dir}\n"
            f"  Mastergrid: {self.mastergrid}\n"
            f"  Mask: {self.mask}\n"
            f"  Constrain: {self.constrain}\n"
            f"  Covariates:\n    {covariate_str}\n"
            f"  Census:\n"
            f"    Path: {self.census['path']}\n"
            f"    Pop Column: {self.census['pop_column']}\n"
            f"    ID Column: {self.census['id_column']}\n"
            f"  Admin Grid:\n{admin_grid_str}\n"
            f"  Processing:\n"
            f"    Log scale: {self.log_scale}\n"
            f"    By Block: {self.by_block}\n"
            f"    Block Size: {self.block_size}\n"
            f"    Max Workers: {self.max_workers}\n"
            f"    Show Progress: {self.show_progress}\n"
            f"    GPU: {self.gpu}\n"
            f"    District Mastergrid: {self.district_dm}\n"
            f"  Logging:\n"
            f"    Level: {self.logging['level']}\n"
            f"    File: {self.logging['file']}\n"
            f"  Output Raster:\n    {output_raster_str if output_raster_str else 'None'}"
        )