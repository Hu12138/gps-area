import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
import alphashape
import matplotlib.pyplot as plt
from pyproj import Transformer
import hdbscan
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class AutoGroupAreaCalculator:
    def __init__(self, min_cluster_size=30, density_radius=5, min_points=20, alpha=0.05,
                 simplify_tolerance=2.0):
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)
        self.min_cluster_size = min_cluster_size
        self.density_radius = density_radius
        self.min_points = min_points
        self.alpha = alpha
        self.simplify_tolerance = simplify_tolerance

    def calculate_work_areas(self, points, visualize=False):
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points])

        # 用HDBSCAN先做大组分组
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,       # 主要调大这个（原300）
            # min_samples=200,             # 次要调大（原5）
            cluster_selection_epsilon=5, # 允许合并5米内的簇（原0）
            cluster_selection_method='eom' # 保持使用EOM方法
            )
        hdb_labels = clusterer.fit_predict(proj_points)
        print(f"一共分了{len(hdb_labels)}组")
        polygons = []
        for label in set(hdb_labels):
            if label == -1:
                continue
            group_points = proj_points[hdb_labels == label]
            if len(group_points) < self.min_points:
                continue

            # 每组用DBSCAN做细聚类
            db = DBSCAN(eps=self.density_radius, min_samples=self.min_points).fit(group_points)
            for db_label in set(db.labels_):
                if db_label == -1:
                    continue
                cluster = group_points[db.labels_ == db_label]

                if len(cluster) < 3:
                    continue

                boundary_points = self._extract_alpha_shape(cluster)
                if len(boundary_points) >= 3:
                    polygons.append(boundary_points)

        areas = []
        valid_polygons = []
        for poly in polygons:
            area = Polygon(poly).area
            if area > 10:
                areas.append(area)
                valid_polygons.append(poly)

        if visualize:
            self._visualize(proj_points, valid_polygons)

        return areas, sum(areas)

    def _extract_alpha_shape(self, points):
        polygon = alphashape.alphashape(points, self.alpha)
        if polygon and polygon.geom_type == 'Polygon':
            simplified = polygon.simplify(self.simplify_tolerance, preserve_topology=True)
            return np.array(simplified.exterior.coords)
        return []

    def _visualize(self, original_points, polygons):
        plt.figure(figsize=(12, 8))
        plt.scatter(original_points[:, 0], original_points[:, 1], c='red', s=30, alpha=0.8, label='原始点')

        for i, poly in enumerate(polygons):
            polygon = Polygon(poly)
            x, y = polygon.exterior.xy
            plt.plot(x, y, linewidth=2, label=f'区域{i + 1}')
            plt.fill(x, y, alpha=0.2)

            centroid = polygon.centroid
            plt.text(centroid.x, centroid.y,
                     f'{polygon.area:.0f}㎡\n({polygon.area / 666.67:.1f}亩)',
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.8))

        plt.title('工作区域识别（HDBSCAN + DBSCAN自动分组）')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()


# 测试调用
if __name__ == "__main__":
    from getData import getData

    test_points = getData("data/13800002122-15.json")

    calculator = AutoGroupAreaCalculator(
        min_cluster_size=30,
        density_radius=5,
        min_points=15,
        alpha=0.3,
        simplify_tolerance=2
    )

    areas, total = calculator.calculate_work_areas(test_points, visualize=True)

    print(f"识别到 {len(areas)} 个工作区域")
    print(f"各区域面积（平方米）: {areas}")
    print(f"总面积: {total:.2f} 平方米 ({total / 666.67:.2f} 亩)")