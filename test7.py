import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import Transformer
import matplotlib.pyplot as plt
from concave_hull import concave_hull
import warnings

# ✅ 设置 macOS 中文字体
plt.rcParams['font.family'] = 'AppleGothic'

class SimpleAreaCalculator:
    def __init__(self, density_radius=15, min_points=8, concave_k=3):
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.density_radius = density_radius
        self.min_points = min_points
        self.concave_k = concave_k

    def calculate_work_areas(self, gps_points, visualize=False):
        # 经纬度 -> 米制投影坐标
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in gps_points])
        print(f"[INFO] 共 {len(proj_points)} 个 GPS 轨迹点")

        clusters = self._density_clustering(proj_points)

        polygons = []
        for i, cluster in enumerate(clusters):
            if len(cluster) < 3:
                continue
            polygon = self._concave_shape(cluster)
            if polygon and polygon.area > 100:
                print(f"[INFO] 第 {i+1} 个区域面积为 {polygon.area:.2f} 平方米")
                polygons.append(polygon)

        areas = [p.area for p in polygons]

        if visualize:
            self._visualize(proj_points, polygons)

        return areas, sum(areas)

    def _density_clustering(self, points):
        db = DBSCAN(eps=self.density_radius, min_samples=self.min_points).fit(points)
        clusters = []
        for label in set(db.labels_):
            if label == -1:
                continue
            cluster = points[db.labels_ == label]
            if len(cluster) >= self.min_points:
                clusters.append(cluster)
        print(f"[INFO] 聚类得到 {len(clusters)} 个簇")
        return clusters

    def _concave_shape(self, points):
        try:
            k = min(self.concave_k, len(points) - 1)
            coords = concave_hull(points.tolist(), k=k)
            return Polygon(coords)
        except Exception as e:
            warnings.warn(f"[WARN] 构建 concave hull 失败: {e}")
            return None

    def _visualize(self, points, polygons):
        plt.figure(figsize=(12, 8))
        plt.scatter(points[:, 0], points[:, 1], c='gray', s=5, alpha=0.4, label='GPS 轨迹点')

        for i, poly in enumerate(polygons):
            x, y = poly.exterior.xy
            plt.plot(x, y, linewidth=2, label=f'区域{i + 1}')
            plt.fill(x, y, alpha=0.2)
            centroid = poly.centroid
            plt.text(centroid.x, centroid.y,
                     f'{poly.area:.0f}㎡\n({poly.area / 666.67:.1f}亩)',
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.8))

        plt.title('耕作区域识别结果', fontsize=14)
        plt.xlabel('投影 X (米)')
        plt.ylabel('投影 Y (米)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

# ✅ 示例使用
if __name__ == "__main__":
    from getData import getData  # 你的 JSON 解析函数
    gps_points = getData("data/test1.json")

    calculator = SimpleAreaCalculator(
        density_radius=15,
        min_points=8,
        concave_k=3
    )
    areas, total = calculator.calculate_work_areas(gps_points, visualize=True)

    print(f"\n识别到 {len(areas)} 个工作区域")
    print(f"各区域面积（平方米）: {[round(a, 2) for a in areas]}")
    print(f"总面积: {total:.2f} 平方米 ({total / 666.67:.2f} 亩)")