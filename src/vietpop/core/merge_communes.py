from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
import itertools
import time

from vietpop.utils.logger import get_logger

logger = get_logger()

class CommuneMerger:
    def __init__(self, settings):
        self.settings = settings
        cfg = settings.merge_communes
        self.shapefile = cfg['shapefile']
        self.district_col = cfg['district_col']
        self.urban_col = cfg['urban_col']
        self.id_col = cfg['id_col']
        self.pop_col = cfg.get('pop_col', 'Total_popu')  # Thêm dòng này để lấy pop_col từ config
        self.target_group_sizes = cfg['target_group_sizes']
        self.urban_restrict_opts = cfg['urban_restrict_opts']
        self.district_restrict_opts = cfg['district_restrict_opts']
        self.outfile = cfg['outfile']
        self.csv_outfile = cfg.get('csv_outfile', 'popcount.csv')

    def get_neighbors(self, gdf):
        sindex = gdf.sindex
        neighbors = {i: set() for i in gdf.index}
        BUFFER = 1e-6  # buffer nhỏ để bắt các polygon gần nhau nhưng không touches chính xác
        for i, geom in gdf.geometry.items():
            buffered = geom.buffer(BUFFER)
            for j_pos in sindex.intersection(buffered.bounds):
                j = gdf.index[j_pos]
                if i != j and (geom.touches(gdf.loc[j].geometry) 
                               or buffered.intersects(gdf.loc[j].geometry)):
                    neighbors[i].add(j)
        return neighbors

    def initial_grouping(self, group, neighbors, target_size, urban_restrict):
        used = set()
        clusters = []
        for idx in group.index:
            if idx in used:
                continue
            cluster = [idx]
            used.add(idx)
            while len(cluster) < target_size:
                best = None
                max_shared = 0
                # ✅ Duyệt tất cả phần tử trong cluster, không chỉ phần tử cuối
                for current in cluster:
                    geom = group.loc[current].geometry
                    urban = group.loc[current, self.urban_col]
                    candidates = [n for n in neighbors[current] if n not in used]
                    if not candidates:
                        continue
                    if urban_restrict:
                        # Ưu tiên cùng urban
                        for n in candidates:
                            shared = geom.boundary.intersection(
                                group.loc[n].geometry.boundary).length
                            if group.loc[n, self.urban_col] == urban and shared > max_shared:
                                max_shared = shared
                                best = n
                        # Fallback: khác urban
                        if best is None:
                            for n in candidates:
                                shared = geom.boundary.intersection(
                                    group.loc[n].geometry.boundary).length
                                if shared > max_shared:
                                    max_shared = shared
                                    best = n
                    else:
                        for n in candidates:
                            shared = geom.boundary.intersection(
                                group.loc[n].geometry.boundary).length
                            if shared > max_shared:
                                max_shared = shared
                                best = n
                if best is None or max_shared == 0:
                    break  # Không còn neighbor nào hợp lệ
                cluster.append(best)
                used.add(best)
            clusters.append(cluster)
        return clusters

    def attach_singletons(self, group, clusters, urban_restrict):
        cluster_list = clusters.copy()
        geom_cache = {}
        iteration = 0
        while True:
            singletons = [c for c in cluster_list if len(c) == 1]
            if not singletons:
                break
            iteration += 1
            logger.info(f"  Singleton iteration {iteration}: {len(singletons)} singletons, {len(cluster_list)} total clusters")
            merged_any = False
            for s in singletons:
                s_idx = s[0]
                s_geom = group.loc[s_idx].geometry
                s_urban = group.loc[s_idx, self.urban_col]
                s_bounds = s_geom.bounds
                same_urban = []
                diff_urban = []
                for c in cluster_list:
                    if c is s:
                        continue
                    c_key = tuple(c)
                    if c_key not in geom_cache:
                        geom_cache[c_key] = unary_union(group.loc[c].geometry)
                    c_geom = geom_cache[c_key]
                    c_bounds = c_geom.bounds
                    if (s_bounds[2] < c_bounds[0] or s_bounds[0] > c_bounds[2] or
                        s_bounds[3] < c_bounds[1] or s_bounds[1] > c_bounds[3]):
                        continue
                    shared = s_geom.boundary.intersection(c_geom.boundary).length
                    dist = s_geom.centroid.distance(c_geom.centroid)
                    rec = {"cluster": c, "shared": shared, "dist": dist}
                    if group.loc[c[0], self.urban_col] == s_urban:
                        same_urban.append(rec)
                    else:
                        diff_urban.append(rec)
                if urban_restrict:
                    boundary_same = [r for r in same_urban if r["shared"] > 0]
                    if boundary_same:
                        target = max(boundary_same, key=lambda r: r["shared"])["cluster"]
                    else:
                        boundary_any = [r for r in diff_urban if r["shared"] > 0]
                        if boundary_any:
                            target = max(boundary_any, key=lambda r: r["shared"])["cluster"]
                        else:
                            all_candidates = same_urban + diff_urban
                            if not all_candidates:
                                continue
                            target = min(all_candidates, key=lambda r: r["dist"])["cluster"]
                else:
                    all_candidates = same_urban + diff_urban
                    if not all_candidates:
                        continue
                    boundary_any = [r for r in all_candidates if r["shared"] > 0]
                    if boundary_any:
                        target = max(boundary_any, key=lambda r: r["shared"])["cluster"]
                    else:
                        target = min(all_candidates, key=lambda r: r["dist"])["cluster"]
                target.append(s_idx)
                cluster_list.remove(s)
                c_key = tuple(target)
                if c_key in geom_cache:
                    del geom_cache[c_key]
                merged_any = True
                break
            if not merged_any:
                break
        return cluster_list

    def attach_small_clusters(self, group, clusters, urban_restrict, target_size):
        """Gộp các cluster nhỏ hơn target_size vào cluster lân cận lớn nhất."""
        cluster_list = clusters.copy()
        geom_cache = {}
        iteration = 0
        while True:
            # ✅ Xử lý tất cả cluster nhỏ hơn target_size, không chỉ singleton
            small = [c for c in cluster_list if len(c) < target_size]
            if not small:
                break
            # Ưu tiên xử lý cluster nhỏ nhất trước
            small.sort(key=lambda c: len(c))
            iteration += 1
            logger.info(
                f"  Small cluster iteration {iteration}: "
                f"{len(small)} small clusters (size < {target_size}), "
                f"{len(cluster_list)} total clusters"
            )
            merged_any = False
            for s in small:
                s_key = tuple(s)
                if s_key not in geom_cache:
                    geom_cache[s_key] = unary_union(group.loc[s].geometry)
                s_geom = geom_cache[s_key]
                s_urban = group.loc[s[0], self.urban_col]
                s_bounds = s_geom.bounds
                same_urban = []
                diff_urban = []
                for c in cluster_list:
                    if c is s:
                        continue
                    c_key = tuple(c)
                    if c_key not in geom_cache:
                        geom_cache[c_key] = unary_union(group.loc[c].geometry)
                    c_geom = geom_cache[c_key]
                    c_bounds = c_geom.bounds
                    # Bounding box check
                    if (s_bounds[2] < c_bounds[0] or s_bounds[0] > c_bounds[2] or
                            s_bounds[3] < c_bounds[1] or s_bounds[1] > c_bounds[3]):
                        continue
                    shared = s_geom.boundary.intersection(c_geom.boundary).length
                    dist = s_geom.centroid.distance(c_geom.centroid)
                    rec = {"cluster": c, "shared": shared, "dist": dist}
                    if group.loc[c[0], self.urban_col] == s_urban:
                        same_urban.append(rec)
                    else:
                        diff_urban.append(rec)
                # Chọn target cluster để gộp vào
                target = None
                if urban_restrict:
                    boundary_same = [r for r in same_urban if r["shared"] > 0]
                    if boundary_same:
                        target = max(boundary_same, key=lambda r: r["shared"])["cluster"]
                    else:
                        boundary_any = [r for r in (same_urban + diff_urban) if r["shared"] > 0]
                        if boundary_any:
                            target = max(boundary_any, key=lambda r: r["shared"])["cluster"]
                        else:
                            all_c = same_urban + diff_urban
                            if all_c:
                                target = min(all_c, key=lambda r: r["dist"])["cluster"]
                else:
                    all_c = same_urban + diff_urban
                    if not all_c:
                        continue
                    boundary_any = [r for r in all_c if r["shared"] > 0]
                    target = (max(boundary_any, key=lambda r: r["shared"])["cluster"]
                              if boundary_any
                              else min(all_c, key=lambda r: r["dist"])["cluster"])
                if target is None:
                    continue
                # Gộp s vào target
                old_key = tuple(target)
                target.extend(s)
                cluster_list.remove(s)
                # Xóa cache của target vì đã thay đổi
                for k in [old_key, tuple(target), s_key]:
                    geom_cache.pop(k, None)
                merged_any = True
                break  # Restart vòng lặp sau mỗi lần gộp
            if not merged_any:
                break
        return cluster_list

    def build_merged_rows(self, group, clusters, district):
        rows = []
        exclude_cols = {self.id_col, 'TYPE_3', 'comuune_le', 'commuune__1'}
        keep_first_cols = {'urban_divi', 'provinci_1', 'district_1'}
        keep_cols = [c for c in group.columns if c not in exclude_cols and c != 'geometry']
        num_cols = group[keep_cols].select_dtypes(include='number').columns.tolist()
        num_cols = [c for c in num_cols if c not in keep_first_cols]
        non_num_cols = [c for c in keep_cols if c not in num_cols and c not in keep_first_cols]
        for idx, cluster in enumerate(clusters):
            geom = unary_union(group.loc[cluster].geometry)
            row = {
                self.id_col: idx + 1,  # Đánh số unique cho id_col
                'district': district,
                'merged_ids': group.loc[cluster, self.id_col].tolist(),
                'geometry': geom
            }
            for col in num_cols:
                row[col] = group.loc[cluster, col].sum()
            for col in keep_first_cols.union(non_num_cols):
                if col in group.columns:
                    row[col] = group.loc[cluster[0], col]
            if 'Shape_Leng' in group.columns:
                row['Shape_Leng'] = geom.length
            if 'Shape_Area' in group.columns:
                row['Shape_Area'] = geom.area
            rows.append(row)
        return rows

    def merge(self):
        gdf = gpd.read_file(self.shapefile)
        gdf = gdf[gdf.is_valid]
        special = gdf[gdf[self.district_col].isna()].copy()
        normal = gdf[~gdf[self.district_col].isna()].copy()
        all_outputs = []
        for tgs, urban_restrict, district_restrict in itertools.product(
            self.target_group_sizes, self.urban_restrict_opts, self.district_restrict_opts
        ):
            fname = self.outfile
            fname = Path(self.outfile).with_stem(
                f"{Path(self.outfile).stem}_gs{tgs}_urban{int(urban_restrict)}_district{int(district_restrict)}"
            )
            csv_fname = Path(self.csv_outfile).with_stem(
                f"{Path(self.csv_outfile).stem}_gs{tgs}_urban{int(urban_restrict)}_district{int(district_restrict)}"
            ).with_suffix('.csv')
            if Path(fname).exists():
                logger.info(f"{fname} already exists, skipping...")
                continue
            logger.info(f"=== Running: group_size={tgs}, urban_restrict={urban_restrict}, district_restrict={district_restrict} ===")
            merged_rows = []
            start_time = time.time()
            if district_restrict:
                for district, group in normal.groupby(self.district_col):
                    logger.info(f"Processing district: {district}")
                    neighbors = self.get_neighbors(group)
                    clusters = self.initial_grouping(group, neighbors, tgs, urban_restrict)
                    clusters = self.attach_small_clusters(group, clusters, urban_restrict, tgs)
                    merged_rows.extend(self.build_merged_rows(group, clusters, district))
            else:
                group = normal.copy()
                logger.info("Processing all communes as one group (no district restriction)")
                logger.info(f"Total features: {len(group)}")
                neighbors = self.get_neighbors(group)
                logger.info("Neighbor detection done.")
                clusters = self.initial_grouping(group, neighbors, tgs, urban_restrict)
                clusters = self.attach_small_clusters(group, clusters, urban_restrict, tgs)
                merged_rows.extend(self.build_merged_rows(group, clusters, 'ALL'))
            merged = gpd.GeoDataFrame(merged_rows, geometry='geometry', crs=gdf.crs)
            final = pd.concat([merged, special], ignore_index=True)
            final = gpd.GeoDataFrame(final, geometry='geometry', crs=gdf.crs)
            # Xóa các dòng null geometry hoặc thiếu id/pop
            final = final[final.geometry.notnull()]
            final = final[final[self.id_col].notnull()]
            final = final[final[self.pop_col].notnull()]
            final = final[final[self.pop_col] != 0]
            final.to_file(fname)
            logger.info(f"✅ {fname} saved successfully")
            # --- Xuất file CSV ---
            # id_col là id, pop_col là pop
            if self.id_col in final.columns and self.pop_col in final.columns:
                csv_df = final[[self.id_col, self.pop_col]].rename(
                    columns={self.id_col: 'id', self.pop_col: 'pop'}
                )
                csv_df.to_csv(csv_fname, index=False)
                logger.info(f"✅ {csv_fname} saved successfully")
            else:
                logger.warning(f"Không tìm thấy cột '{self.id_col}' hoặc '{self.pop_col}' trong kết quả để xuất CSV.")
            logger.info(f"⏱️ Elapsed time: {time.time() - start_time:.2f} seconds")
            all_outputs.append(fname)
        return all_outputs