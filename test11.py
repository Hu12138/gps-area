import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
import alphashape
import matplotlib.pyplot as plt
import matplotlib
from pyproj import Transformer


matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class OptimizedAreaCalculator:
    def __init__(self, density_radius=5, min_points=20, alpha=0.05,
                 simplify_tolerance=2.0, offset_distance=3):
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)
        self.density_radius = density_radius
        self.min_points = min_points
        self.alpha = alpha
        self.simplify_tolerance = simplify_tolerance
        self.offset_distance = offset_distance  # 偏移扩展的距离（米）

    def calculate_work_areas(self, points, visualize=False):
        # 坐标转换（经纬度 -> 米）
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points])

        # 原始点 + 上下左右扩展点
        extended_points = self._add_surrounding_points(proj_points)

        # 聚类分析
        clusters = self._density_clustering(extended_points)

        # 提取边界轮廓
        polygons = []
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            boundary_points = self._extract_alpha_shape(cluster)
            if len(boundary_points) >= 3:
                polygons.append(boundary_points)

        # 面积计算
        areas = []
        valid_polygons = []
        for poly in polygons:
            area = Polygon(poly).area
            if area > 10:
                areas.append(area)
                valid_polygons.append(poly)

        if visualize:
            self._visualize(proj_points, extended_points, valid_polygons)

        return areas, sum(areas)

    def _add_surrounding_points(self, points):
        # 对每个点增加 4 个方向的偏移点（上、下、左、右）
        offsets = np.array([
            [0, self.offset_distance],    # 上（北）
            [0, -self.offset_distance],   # 下（南）
            [-self.offset_distance, 0],   # 左（西）
            [self.offset_distance, 0]     # 右（东）
        ])

        extended = []
        for p in points:
            extended.append(p)
            for offset in offsets:
                extended.append(p + offset)
        return np.array(extended)

    def _density_clustering(self, points):
        db = DBSCAN(eps=self.density_radius, min_samples=self.min_points).fit(points)
        clusters = []
        for label in set(db.labels_):
            if label == -1:
                continue
            cluster = points[db.labels_ == label]
            clusters.append(cluster)
        return clusters

    def _extract_alpha_shape(self, points):
        polygon = alphashape.alphashape(points, self.alpha)
        if polygon and polygon.geom_type == 'Polygon':
            simplified = polygon.simplify(self.simplify_tolerance, preserve_topology=True)
            return np.array(simplified.exterior.coords)
        return []

    def _visualize(self, original_points, extended_points, polygons):
        plt.figure(figsize=(12, 8))

        plt.scatter(original_points[:, 0], original_points[:, 1],
                    c='red', s=30, alpha=0.8, label='原始点')
        plt.scatter(extended_points[:, 0], extended_points[:, 1],
                    c='gray', s=5, alpha=0.4, label='扩展点')

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

        plt.title('工作区域识别（仅上下左右扩展）')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from getData import getData
    # test_points = getData("data/2134.json")
    test_points = getData("data/test1.json")
    # test_points = getData("data/bug.json")
    # test_points = getData("data/13800002122-14.json")
    # test_points = getData("data/13800002122-15.json")
    # test_points = getData("data/13885004840-11.json")

    calculator = OptimizedAreaCalculator(
        density_radius=2,
        min_points=10,
        alpha=0.3,
        offset_distance=1  # 上下左右扩展2米
    )

    areas, total = calculator.calculate_work_areas(test_points, visualize=True)

    print(f"识别到 {len(areas)} 个工作区域")
    print(f"各区域面积（平方米）: {areas}")
    print(f"总面积: {total:.2f} 平方米 ({total / 666.67:.2f} 亩)")