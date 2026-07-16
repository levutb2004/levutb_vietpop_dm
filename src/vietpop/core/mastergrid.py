# src/vietpop/core/mastergrid.py
from pathlib import Path
from typing import Optional

import geopandas as gpd
import rasterio

from ..config.settings import Settings
from ..utils.logger import get_logger
from ..utils.vector import rasterize

logger = get_logger()


class MastergridCreator:
    """
    Create mastergrid raster from a shapefile or use existing one.

    This class handles the creation of a mastergrid raster from an
    administrative shapefile, or loads an existing mastergrid if already
    provided in settings.

    Attributes:
        settings (Settings): Configuration settings
        output_dir (Path): Directory for output files

    Example:
        >>> creator = MastergridCreator(settings)
        >>> mastergrid_path = creator.create()
    """

    def __init__(self, settings: Settings):
        """
        Initialize MastergridCreator.

        Args:
            settings: Configuration settings instance
        """
        self.settings = settings
        self.output_dir = Path(settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"MastergridCreator initialized. Output dir: {self.output_dir}")

    def _get_template(self) -> Optional[str]:
        """
        Get template raster path from first available covariate.

        Returns:
            Path to template raster, or None if no covariates available

        Raises:
            ValueError: If no covariates are configured
        """
        if not self.settings.covariate:
            raise ValueError(
                "No covariates configured. At least one covariate is required "
                "as a spatial template when creating mastergrid from shapefile."
            )
        # Use first covariate as template
        template_key = next(iter(self.settings.covariate))
        template_path = self.settings.covariate[template_key]
        logger.debug(f"Using covariate '{template_key}' as spatial template: {template_path}")
        return template_path

    def _validate_shapefile(self, shp_path: str, column: str) -> gpd.GeoDataFrame:
        """
        Load and validate the admin shapefile.

        Args:
            shp_path: Path to shapefile
            column: Column name to use as zone ID

        Returns:
            Validated GeoDataFrame

        Raises:
            FileNotFoundError: If shapefile does not exist
            ValueError: If column not found or geometries invalid
        """
        path = Path(shp_path)
        if not path.exists():
            raise FileNotFoundError(f"Admin shapefile not found: {shp_path}")

        logger.info(f"Loading shapefile: {shp_path}")
        gdf = gpd.read_file(shp_path)
        logger.info(f"Loaded {len(gdf)} features, CRS: {gdf.crs}")

        if column not in gdf.columns:
            available = list(gdf.columns)
            raise ValueError(
                f"Column '{column}' not found in shapefile. "
                f"Available columns: {available}"
            )

        # Check for null values in ID column
        null_count = gdf[column].isnull().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null values in column '{column}'. They will be excluded.")
            gdf = gdf[gdf[column].notna()].copy()

        # Check for duplicate IDs
        dup_count = gdf[column].duplicated().sum()
        if dup_count > 0:
            logger.warning(f"Found {dup_count} duplicate values in column '{column}'.")

        # Validate geometries
        invalid = (~gdf.geometry.is_valid).sum()
        if invalid > 0:
            logger.warning(f"Found {invalid} invalid geometries. Attempting to fix...")
            gdf.geometry = gdf.geometry.buffer(0)

        logger.info(f"Shapefile validated: {len(gdf)} valid features")
        logger.debug(f"ID column '{column}' range: [{gdf[column].min()}, {gdf[column].max()}]")
        return gdf

    def _check_crs_match(self, gdf: gpd.GeoDataFrame, template_path: str) -> gpd.GeoDataFrame:
        """
        Ensure shapefile CRS matches template raster CRS, reproject if needed.

        Args:
            gdf: Input GeoDataFrame
            template_path: Path to template raster

        Returns:
            GeoDataFrame with matching CRS
        """
        with rasterio.open(template_path) as src:
            raster_crs = src.crs

        if gdf.crs is None:
            logger.warning("Shapefile has no CRS defined. Assuming it matches the template.")
            return gdf

        if gdf.crs != raster_crs:
            logger.warning(
                f"CRS mismatch: shapefile={gdf.crs}, template={raster_crs}. "
                f"Reprojecting shapefile..."
            )
            gdf = gdf.to_crs(raster_crs)
            logger.info("Reprojection completed.")
        else:
            logger.debug("CRS match confirmed.")

        return gdf

    def create_from_shapefile(self,
                              shp_path: str,
                              column: str = 'OBJECTID',
                              outfile: Optional[str] = None) -> str:
        """
        Create mastergrid raster from admin shapefile.

        Args:
            shp_path: Path to admin shapefile
            column: Column name in shapefile to use as zone ID values
            outfile: Output path for mastergrid raster.
                     Defaults to output_dir/mastergrid.tif

        Returns:
            Path to created mastergrid raster

        Raises:
            FileNotFoundError: If shapefile or template not found
            ValueError: If inputs are invalid
            RuntimeError: If rasterization fails
        """
        logger.info("=" * 50)
        logger.info("Creating mastergrid from shapefile")
        logger.info(f"  Source : {shp_path}")
        logger.info(f"  Column : {column}")
        logger.info("=" * 50)

        # Resolve output path
        if outfile is None:
            outfile = str(self.output_dir / 'mastergrid.tif')
        outfile = str(Path(outfile).resolve())

        # Skip if already exists
        if Path(outfile).exists():
            logger.info(f"Mastergrid already exists, skipping creation: {outfile}")
            return outfile

        # Get template
        template_path = self._get_template()

        # Validate and load shapefile
        gdf = self._validate_shapefile(shp_path, column)

        # Align CRS
        gdf = self._check_crs_match(gdf, template_path)

        # Rasterize
        logger.info("Starting rasterization...")
        try:
            rasterize(
                source=gdf,
                outfile=outfile,
                template=template_path,
                column=column,
                dtype='int32',
                by_block=True,
                max_workers=self.settings.max_workers,
                show_progress=self.settings.show_progress,
                block_size=self.settings.block_size
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create mastergrid from shapefile: {str(e)}")

        # Verify output
        if not Path(outfile).exists():
            raise RuntimeError(f"Rasterization completed but output file not found: {outfile}")

        with rasterio.open(outfile) as src:
            logger.info("Mastergrid created successfully:")
            logger.info(f"  Path      : {outfile}")
            logger.info(f"  Shape     : {src.height} x {src.width}")
            logger.info(f"  CRS       : {src.crs}")
            logger.info(f"  Transform : {src.transform}")
            logger.info(f"  NoData    : {src.nodata}")

        return outfile

    def create(self) -> str:
        """
        Main entry point. Use existing mastergrid or create from shapefile.

        If settings.mastergrid is already set and the file exists,
        it is returned directly. Otherwise, creates a new mastergrid
        from settings.admin shapefile.

        Returns:
            Path to mastergrid raster

        Raises:
            ValueError: If neither mastergrid nor admin shapefile is configured
            FileNotFoundError: If configured files are not found
            RuntimeError: If mastergrid creation fails
        """
        # Case 1: mastergrid already exists
        if self.settings.mastergrid and self.settings.mastergrid != 'create':
            mastergrid_path = Path(self.settings.mastergrid)
            if mastergrid_path.exists():
                logger.info(f"Using existing mastergrid: {self.settings.mastergrid}")
                return str(self.settings.mastergrid)
            else:
                logger.warning(
                    f"Mastergrid path configured but file not found: {self.settings.mastergrid}. "
                    f"Will attempt to create from shapefile."
                )

        # Case 2: create from shapefile
        if not self.settings.admin:
            raise ValueError(
                "No mastergrid file found and no admin shapefile configured. "
                "Please set 'mastergrid' or 'admin' in config.yaml."
            )

        logger.info("No existing mastergrid found. Creating from admin shapefile...")

        # Ưu tiên outfile từ config nếu có
        if self.settings.admin_grid_outfile:
            outfile = str(Path(self.settings.admin_grid_outfile))
            if not Path(outfile).is_absolute():
                outfile = str(Path(self.settings.output_dir) / self.settings.admin_grid_outfile)
        else:
            outfile = str(self.output_dir / 'mastergrid.tif')

        mastergrid_path = self.create_from_shapefile(
            shp_path=self.settings.admin,
            column=self.settings.admin_column,
            outfile=outfile
        )

        return mastergrid_path