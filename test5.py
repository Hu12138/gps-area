import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
from pyproj import Transformer
import matplotlib.pyplot as plt
from shapely.geometry import LineString

class SimpleAreaCalculator:
    def __init__(self, density_radius=5, min_points=20):
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.density_radius = density_radius  # 单位：米（投影坐标系）
        self.min_points = min_points

    def calculate_work_areas(self, points, visualize=False):
        """三步式计算工作区域"""
        # 第一步：密度聚类（模仿人眼观察）
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points])
        clusters = self._density_clustering(proj_points)
        
        # 第二步：边界提取（模仿人类画线）
        polygons = []
        for cluster in clusters:
            if len(cluster) < 3:  # 至少3个点才能形成面
                continue
            hull = ConvexHull(cluster)
            boundary_points = cluster[hull.vertices]  # 获取凸包边界点
            polygons.append(self._simplify_boundary(boundary_points))
        
        # 第三步：面积计算（模仿地图工具）
        areas = []
        valid_polygons = []
        for poly in polygons:
            area = Polygon(poly).area
            if area > 100:  # 过滤极小区域（100平方米）
                areas.append(area)
                valid_polygons.append(poly)
        
        if visualize:
            self._visualize(proj_points, valid_polygons)
        
        return areas, sum(areas)

    def _density_clustering(self, points):
        """密度聚类（核心步骤）"""
        db = DBSCAN(eps=self.density_radius, min_samples=self.min_points).fit(points)
        
        clusters = []
        for label in set(db.labels_):
            if label == -1:
                continue  # 忽略噪声点
            cluster = points[db.labels_ == label]
            if len(cluster) >= self.min_points:
                clusters.append(cluster)
        return clusters

    def _simplify_boundary(self, points, tolerance=2.0):
        """边界简化（Ramer-Douglas-Peucker算法）"""
        line = LineString(points)
        simplified = line.simplify(tolerance, preserve_topology=True)
        return np.array(simplified.coords)

    def _visualize(self, points, polygons):
        """可视化结果"""
        plt.figure(figsize=(12, 8))
        
        # 绘制原始点
        plt.scatter(points[:,0], points[:,1], c='gray', s=5, alpha=0.3, label='原始点')
        
        # 绘制多边形
        for i, poly in enumerate(polygons):
            polygon = Polygon(poly)
            x, y = polygon.exterior.xy
            plt.plot(x, y, linewidth=2, label=f'区域{i+1}')
            plt.fill(x, y, alpha=0.2)
            
            # 标注面积
            centroid = polygon.centroid
            plt.text(centroid.x, centroid.y, 
                    f'{polygon.area:.0f}㎡\n({polygon.area/666.67:.1f}亩)',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title('工作区域识别结果', fontsize=14)
        plt.xlabel('投影坐标 X (米)', fontsize=10)
        plt.ylabel('投影坐标 Y (米)', fontsize=10)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

# 测试示例
if __name__ == "__main__":
    # 生成测试数据（实际应替换为真实GPS数据）
    def generate_test_field():
        # 生成一个矩形区域（约3000平方米）
        x = np.linspace(116.400, 116.405, 30)
        y = np.linspace(39.900, 39.903, 30)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])
    from getData import getData
    test_points = getData("data/test1.json")
    
    calculator = SimpleAreaCalculator(density_radius=5, min_points=10)
    areas, total = calculator.calculate_work_areas(test_points, visualize=True)
    
    print(f"识别到 {len(areas)} 个工作区域")
    print(f"各区域面积（平方米）: {areas}")
    print(f"总面积: {total:.2f} 平方米 ({total/666.67:.2f} 亩)")