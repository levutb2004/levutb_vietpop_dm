import numpy as np
import pandas as pd
import rasterio
from typing import Optional, Any
from scipy.stats import skew, kurtosis
from libpysal.weights import Queen, W
from esda.moran import Moran, Moran_Local
from pathlib import Path
from ..utils.raster import raster_stat
from ..utils.logger import get_logger
logger = get_logger()

class ResidualDiagnostics:
    """
    Container for residuals and their spatial/statistical diagnostics.
    """
    def __init__(self, data: pd.DataFrame, dens_pred: np.ndarray, geometry: Any):
        self.geometry = geometry
        self.data = data
        self.data['dens_pred'] = dens_pred

         # Compute Queen spatial weights from geometry
        self.spatial_weights_original = Queen.from_dataframe(self.geometry)
        self.spatial_weights, self.data_filtered = self._filter_island()

        self.dens_true = self.data_filtered['dens'].values
        self.dens_pred = self.data_filtered['dens_pred'].values
        self.residuals = self.dens_true - self.dens_pred

       
        # Filter isolated zones (no neighbors)
        # self.residuals, self.y_true, self.y_pred, self.data_filtered, self.spatial_weights = self._filter_isolated()
    
    def _filter_island(self):
        gdf = self.geometry
        data_indexed = self.data.set_index('id')
        data_filtered = data_indexed.reindex(gdf['OBJECTID']).dropna(subset=['dens', 'dens_pred'])
        
        w = self.spatial_weights_original
        if len(data_filtered) != len(gdf):
            gdf = gdf[gdf['OBJECTID'].isin(data_filtered.index)].reset_index(drop=True)
            w = Queen.from_dataframe(gdf)
            data_filtered = data_filtered.reindex(gdf['OBJECTID'])
        
        return w, data_filtered

    def _filter_isolated(self):
        """
        Loại bỏ các vùng không có neighbor khỏi residuals, y_true, y_pred, data.
        Trả về tuple (residuals, y_true, y_pred, data, spatial_weights)
        """
        neighbors = self.spatial_weights_original.neighbors
        valid_ids = [i for i, neigh in neighbors.items() if len(neigh) > 0]
        if len(valid_ids) == 0:
            return self.residuals_original, self.y_true_original, self.y_pred_original, self.data, self.spatial_weights_original

        residuals = np.array(self.residuals_original)[valid_ids]
        y_true = np.array(self.y_true_original)[valid_ids]
        y_pred = np.array(self.y_pred_original)[valid_ids]
        data_filtered = self.data.iloc[valid_ids].reset_index(drop=True)

        valid_weights = W({i: [n for n in neighbors[i] if n in valid_ids] for i in valid_ids})

        return residuals, y_true, y_pred, data_filtered, valid_weights

    # --- Basic statistics ---
    @property
    def mean(self): return float(np.mean(self.residuals))
    @property
    def std(self): return float(np.std(self.residuals))
    @property
    def skewness(self): return float(skew(self.residuals))
    @property
    def kurtosis(self): return float(kurtosis(self.residuals))
    @property
    def min(self): return float(np.min(self.residuals))
    @property
    def max(self): return float(np.max(self.residuals))
    @property
    def q25(self): return float(np.percentile(self.residuals, 25))
    @property
    def median(self): return float(np.median(self.residuals))
    @property
    def q75(self): return float(np.percentile(self.residuals, 75))

    # --- Regression metrics ---
    @property
    def rmse(self): return float(np.sqrt(np.mean(self.residuals ** 2)))
    @property
    def mae(self): return float(np.mean(np.abs(self.residuals)))
    @property
    def sse(self): return float(np.sum(self.residuals ** 2))
    @property
    def r2(self): 
        ss_tot = np.sum((self.dens_true - np.mean(self.dens_true)) ** 2)
        return 1 - self.sse / ss_tot if ss_tot > 0 else np.nan

    # --- Spatial statistics ---
    @property
    def moran_global(self):
        if Queen and Moran and self.spatial_weights is not None:
            m = Moran(self.residuals, self.spatial_weights)
            return {
                'I': m.I,
                'p_sim': m.p_sim,
                'z_norm': m.z_norm,
                'z_rand': m.z_rand
            }
        return None

    @property
    def moran_local(self):
        if Queen and Moran_Local and self.spatial_weights is not None:
            ml = Moran_Local(self.residuals, self.spatial_weights)
            stats = {
                'mean': float(np.mean(ml.Is)),
                'std': float(np.std(ml.Is)),
                'skewness': float(skew(ml.Is)),
                'kurtosis': float(kurtosis(ml.Is)),
                'min': float(np.min(ml.Is)),
                'max': float(np.max(ml.Is)),
                'q25': float(np.percentile(ml.Is, 25)),
                'median': float(np.median(ml.Is)),
                'q75': float(np.percentile(ml.Is, 75)),
                'z_mean': float(np.mean(ml.z_sim)),
                'z_std': float(np.std(ml.z_sim)),
                'z_skewness': float(skew(ml.z_sim)),
                'z_kurtosis': float(kurtosis(ml.z_sim)),
                'z_min': float(np.min(ml.z_sim)),
                'z_max': float(np.max(ml.z_sim)),
                'z_q25': float(np.percentile(ml.z_sim, 25)),
                'z_median': float(np.median(ml.z_sim)),
                'z_q75': float(np.percentile(ml.z_sim, 75)),
            }
            return stats
        return None




class CommPopDiagnostics:
    """
    Evaluate population mapping results at the commune level.
    """
    def __init__(self, settings):
        self.settings = settings

    def evaluate(self):
        logger.info("CommPopDiagnostics: Starting evaluation...")

        # 1. Load raster dasymetric
        dasymetric_path = self.settings.output_raster.get('dasymetric')
        if not dasymetric_path or not Path(dasymetric_path).is_file():
            logger.error(f"Dasymetric raster not found: {dasymetric_path}")
            raise FileNotFoundError(f"Dasymetric raster not found: {dasymetric_path}")

        # 2. Load master_grid
        mastergrid_path = self.settings.mastergrid
        if not mastergrid_path or not Path(mastergrid_path).is_file():
            logger.error(f"Mastergrid not found: {mastergrid_path}")
            raise FileNotFoundError(f"Mastergrid not found: {mastergrid_path}")

        # 3. Gọi raster_stat
        logger.info("CommPopDiagnostics: Calculating zonal statistics...")
        df_dm_stat = raster_stat(
            infile=dasymetric_path,
            mastergrid=mastergrid_path,
            by_block=self.settings.by_block,
            max_workers=self.settings.max_workers,
            block_size=self.settings.block_size,
            show_progress=self.settings.show_progress
        )

        logger.info(f"CommPopDiagnostics: Total zones: {len(df_dm_stat)}")
        df_dm_stat = df_dm_stat[df_dm_stat['sum'] != 0]
        logger.info(f"CommPopDiagnostics: Number of zones after filtering sum=0: {len(df_dm_stat)}")

        # 4. Join với census
        census_path = self.settings.census['path']
        id_col = self.settings.census['id_column']
        pop_col = self.settings.census['pop_column']

        logger.info(f"CommPopDiagnostics: Reading census from {census_path}")
        if not census_path or not Path(census_path).is_file():
            logger.error(f"Census file not found: {census_path}")
            raise FileNotFoundError(f"Census file not found: {census_path}")

        df_census = pd.read_csv(census_path, dtype={id_col: df_dm_stat['id'].dtype})
        logger.info(f"CommPopDiagnostics: Total census zones: {len(df_census)}")
        df_census = df_census[df_census[pop_col] != 0]
        logger.info(f"CommPopDiagnostics: Number of census zones after filtering pop=0: {len(df_census)}")

        df_merge = pd.merge(
            df_dm_stat[['id', 'sum']],
            df_census[[id_col, pop_col]],
            left_on='id', right_on=id_col, how='inner'
        )
        logger.info(f"CommPopDiagnostics: Number of zones after join: {len(df_merge)}")

        # Sau khi merge, lưu ra csv
        output_csv = Path(self.settings.output_dir) / "commpopdiag.csv"
        df_merge.to_csv(output_csv, index=False)
        logger.info(f"CommPopDiagnostics: Output saved to {output_csv}")

        # 5. Tính các chỉ số
        y_true = df_merge[pop_col].values
        y_pred = df_merge['sum'].values
        n = len(y_true)
        rmsd = np.sqrt(np.mean((y_pred - y_true) ** 2))
        percent_rmsd = 100 * rmsd / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
        mae = np.mean(np.abs(y_pred - y_true))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        logger.info(f"CommPopDiagnostics: RMSD={rmsd:.2f}, PercentRMSD={percent_rmsd:.2f}, MAD={mad:.2f}, R2={r2:.4f}, n={n}")

        return {
            'RMSD': rmsd,
            'PercentRMSD': percent_rmsd,
            'MAE': mae,
            'R2': r2,
            'n': n,
            'csv': str(output_csv)
        }
class PixelPopDiagnostics:
    """
    Evaluate population mapping results at the pixel level,
    including prediction-interval (PI) diagnostics.
    """
    def __init__(self, settings):
        self.settings = settings

    def evaluatePI(self):
        logger.info("PixelPopDiagnostics: Starting evaluation...")

        # 1. Lấy đường dẫn raster từ settings (đã có sẵn: ground_truth, mean, lower, upper)
        gt_path = self.settings.ground_truth
        mean_path = self.settings.output_raster.get('mean')
        lower_path = self.settings.output_raster.get('lower')   # percentile 2.5
        upper_path = self.settings.output_raster.get('upper')   # percentile 97.5

        for name, p in [('ground_truth', gt_path), ('mean', mean_path),
                         ('lower', lower_path), ('upper', upper_path)]:
            if not p or not Path(p).is_file():
                logger.error(f"{name} raster not found: {p}")
                raise FileNotFoundError(f"{name} raster not found: {p}")

        # 2. Đọc raster pixel-level
        logger.info("PixelPopDiagnostics: Reading rasters...")
        with rasterio.open(gt_path) as src:
            y_true = src.read(1).astype(float)
            nodata_gt = src.nodata
        with rasterio.open(mean_path) as src:
            y_pred = src.read(1).astype(float)
            nodata_pred = src.nodata
        with rasterio.open(lower_path) as src:
            lower = src.read(1).astype(float)
        with rasterio.open(upper_path) as src:
            upper = src.read(1).astype(float)

        if not (y_true.shape == y_pred.shape == lower.shape == upper.shape):
            logger.error("PixelPopDiagnostics: Raster shapes do not match")
            raise ValueError("Raster shapes do not match")

        logger.info(f"PixelPopDiagnostics: Total pixels: {y_true.size}")

        # 3. Lọc nodata / pixel không hợp lệ
        mask = np.ones_like(y_true, dtype=bool)
        if nodata_gt is not None:
            mask &= (y_true != nodata_gt)
        if nodata_pred is not None:
            mask &= (y_pred != nodata_pred)
        mask &= ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(lower) & ~np.isnan(upper)

        y_true = y_true[mask]
        y_pred = y_pred[mask]
        lower = lower[mask]
        upper = upper[mask]
        n = y_true.size
        logger.info(f"PixelPopDiagnostics: Number of valid pixels after filtering: {n}")

        # 4. Các chỉ số accuracy cơ bản
        rmsd = np.sqrt(np.mean((y_pred - y_true) ** 2))
        percent_rmsd = 100 * rmsd / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
        mae = np.mean(np.abs(y_pred - y_true))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        
        within_pi = (y_true >= lower) & (y_true <= upper)
        above_upper = y_true > upper
        below_lower = y_true < lower

        pct_pi_correct = float(np.mean(within_pi)) * 100
        pct_above_upper = float(np.mean(above_upper)) * 100
        pct_below_lower = float(np.mean(below_lower)) * 100
        
        metrics = {
                    'RMSD': rmsd,
                    'PercentRMSD': percent_rmsd,
                    'MAE': mae,
                    'R2': r2,
                    'pct_pi_correct': pct_pi_correct,
                    'pct_above_upper': pct_above_upper,
                    'pct_below_lower': pct_below_lower,
                    'n': n,
                }

        # Lưu các chỉ số ra csv
        output_csv = Path(self.settings.output_dir) / "pixelpopdiag.csv"
        pd.DataFrame([metrics]).to_csv(output_csv, index=False)
        logger.info(f"PixelPopDiagnostics: Output saved to {output_csv}")
        
        logger.info(
            f"PixelPopDiagnostics: RMSD={rmsd:.2f}, PercentRMSD={percent_rmsd:.2f}, "
            f"MAE={mae:.2f}, R2={r2:.4f},"
            f"PI_correct={pct_pi_correct:.2f}%, AboveUpper={pct_above_upper:.2f}%, "
            f"BelowLower={pct_below_lower:.2f}%, n={n}"
        )

        return {**metrics, 'csv': str(output_csv)}
    def evaluate(self):
        logger.info("PixelPopDiagnostics: Starting evaluation...")

        # 1. Input rasters
        gt_path = self.settings.ground_truth
        pred_path = self.settings.output_raster.get('dasymetric')

        for name, p in [("ground_truth", gt_path), ("prediction", pred_path)]:
            if not p or not Path(p).is_file():
                logger.error(f"{name} raster not found: {p}")
                raise FileNotFoundError(f"{name} raster not found: {p}")

        # 2. Read rasters
        logger.info("PixelPopDiagnostics: Reading rasters...")

        with rasterio.open(gt_path) as src:
            y_true = src.read(1).astype(np.float32)
            nodata_gt = src.nodata

        with rasterio.open(pred_path) as src:
            y_pred = src.read(1).astype(np.float32)
            nodata_pred = src.nodata

        if y_true.shape != y_pred.shape:
            logger.error("PixelPopDiagnostics: Raster shapes do not match")
            raise ValueError("Raster shapes do not match")

        logger.info(f"PixelPopDiagnostics: Total pixels: {y_true.size}")

        # 3. Filter invalid pixels
        mask = np.ones_like(y_true, dtype=bool)

        if nodata_gt is not None:
            mask &= (y_true != nodata_gt)

        if nodata_pred is not None:
            mask &= (y_pred != nodata_pred)

        mask &= np.isfinite(y_true)
        mask &= np.isfinite(y_pred)

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        n = len(y_true)

        if n == 0:
            raise ValueError("No valid pixels found after filtering.")

        logger.info(f"PixelPopDiagnostics: Number of valid pixels: {n}")

        # 4. Accuracy metrics
        diff = y_pred - y_true

        rmsd = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(np.abs(diff))

        mean_true = np.mean(y_true)
        percent_rmsd = 100 * rmsd / mean_true if mean_true != 0 else np.nan

        ss_res = np.sum(diff ** 2)
        ss_tot = np.sum((y_true - mean_true) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        metrics = {
            "RMSD": rmsd,
            "PercentRMSD": percent_rmsd,
            "MAE": mae,
            "R2": r2,
            "n": n,
        }

        # 5. Save metrics
        output_csv = Path(self.settings.output_dir) / "pixelpopdiag.csv"
        pd.DataFrame([metrics]).to_csv(output_csv, index=False)

        logger.info(f"PixelPopDiagnostics: Output saved to {output_csv}")
        logger.info(
            f"PixelPopDiagnostics: "
            f"RMSD={rmsd:.2f}, "
            f"PercentRMSD={percent_rmsd:.2f}, "
            f"MAE={mae:.2f}, "
            f"R2={r2:.4f}, "
            f"n={n}"
        )

        return {
            **metrics,
            "csv": str(output_csv),
        }