import numpy as np
from scipy.spatial import ConvexHull  # 不再使用但保留以防扩展
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, LineString
from shapely.ops import transform
from pyproj import Transformer
import matplotlib
import matplotlib.pyplot as plt

import alphashape



# ✅ 设置支持中文字体（适用于 macOS 和常见平台）
matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class SimpleAreaCalculator:
    def __init__(self, density_radius=5, min_points=20, alpha=0.05, simplify_tolerance=2.0):
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.density_radius = density_radius  # 单位：米
        self.min_points = min_points
        self.alpha = alpha
        self.simplify_tolerance = simplify_tolerance

    def calculate_work_areas(self, points, visualize=False):
        """三步式计算工作区域"""
        # 第一步：密度聚类
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points])
        clusters = self._density_clustering(proj_points)
        
        # 第二步：边界提取（使用 alpha shape 替代 convex hull）
        polygons = []
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            boundary_points = self._extract_alpha_shape(cluster)
            if len(boundary_points) >= 3:
                polygons.append(boundary_points)
        
        # 第三步：面积计算
        areas = []
        valid_polygons = []
        for poly in polygons:
            area = Polygon(poly).area
            if area > 100:  # 过滤极小区域
                areas.append(area)
                valid_polygons.append(poly)
        
        if visualize:
            self._visualize(proj_points, valid_polygons)
        
        return areas, sum(areas)

    def _density_clustering(self, points):
        """密度聚类"""
        db = DBSCAN(eps=self.density_radius, min_samples=self.min_points).fit(points)
        clusters = []
        for label in set(db.labels_):
            if label == -1:
                continue
            cluster = points[db.labels_ == label]
            if len(cluster) >= self.min_points:
                clusters.append(cluster)
        return clusters

    def _extract_alpha_shape(self, points):
        """提取凹边界（Alpha Shape + 简化）"""
        polygon = alphashape.alphashape(points, self.alpha)
        if polygon and polygon.geom_type == 'Polygon':
            simplified = polygon.simplify(self.simplify_tolerance, preserve_topology=True)
            return np.array(simplified.exterior.coords)
        return []

    def _visualize(self, points, polygons):
        """可视化结果"""
        plt.figure(figsize=(12, 8))
        
        plt.scatter(points[:, 0], points[:, 1], c='gray', s=5, alpha=0.3, label='原始点')
        
        for i, poly in enumerate(polygons):
            polygon = Polygon(poly)
            x, y = polygon.exterior.xy
            plt.plot(x, y, linewidth=2, label=f'区域{i+1}')
            plt.fill(x, y, alpha=0.2)

            centroid = polygon.centroid
            plt.text(centroid.x, centroid.y,
                     f'{polygon.area:.0f}㎡\n({polygon.area / 666.67:.1f}亩)',
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title('工作区域识别结果', fontsize=14)
        plt.xlabel('投影坐标 X (米)', fontsize=10)
        plt.ylabel('投影坐标 Y (米)', fontsize=10)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    from getData import getData
    test_points = getData("data/test1.json")

    calculator = SimpleAreaCalculator(density_radius=5, min_points=10, alpha=0.4,simplify_tolerance = 2)
    areas, total = calculator.calculate_work_areas(test_points, visualize=True)

    print(f"识别到 {len(areas)} 个工作区域")
    print(f"各区域面积（平方米）: {areas}")
    print(f"总面积: {total:.2f} 平方米 ({total / 666.67:.2f} 亩)")