import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
import alphashape
import matplotlib.pyplot as plt
import matplotlib
from pyproj import Transformer


# ✅ 设置支持中文字体（适用于 macOS 和常见平台）
matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class OptimizedAreaCalculator:
    def __init__(self, density_radius=5, min_points=20, alpha=0.05, 
                 simplify_tolerance=2.0, interpolation_threshold=10):
        # 使用EPSG:3410等面积投影
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)
        self.density_radius = density_radius  # 单位：米
        self.min_points = min_points
        self.alpha = alpha
        self.simplify_tolerance = simplify_tolerance
        self.interpolation_threshold = interpolation_threshold  # 点间距超过此值(米)时触发插值

    def calculate_work_areas(self, points, visualize=False):
        """优化后的工作区域计算"""
        # 坐标转换
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points])
        
        # 自适应插值
        interpolated_points = self._adaptive_interpolation(proj_points)
        
        # 密度聚类
        clusters = self._density_clustering(interpolated_points)
        
        # 边界提取
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
            if area > 10:  # 过滤极小区域
                areas.append(area)
                valid_polygons.append(poly)
        
        if visualize:
            self._visualize(proj_points, interpolated_points, valid_polygons)
        
        return areas, sum(areas)

    def _adaptive_interpolation(self, points):
        """自适应线性插值"""
        if len(points) < 2:
            return points
        
        # 计算点间距
        dist_matrix = distance_matrix(points, points)
        np.fill_diagonal(dist_matrix, np.inf)
        min_distances = np.min(dist_matrix, axis=1)
        
        # 找出需要插值的线段
        segments_to_interpolate = []
        for i in range(len(points)-1):
            dist = np.linalg.norm(points[i] - points[i+1])
            if dist > self.interpolation_threshold:
                segments_to_interpolate.append((i, i+1, dist))
        
        # 如果没有需要插值的线段，直接返回
        if not segments_to_interpolate:
            return points
        
        # 进行插值
        interpolated = [points[0]]
        for i in range(1, len(points)):
            # 检查前一段是否需要插值
            needs_interp = any(start == i-1 for start, _, _ in segments_to_interpolate)
            
            if needs_interp:
                # 找到对应的线段
                segment = next(s for s in segments_to_interpolate if s[0] == i-1)
                start_idx, end_idx, dist = segment
                
                # 计算需要插入的点数
                num_points_to_insert = int(dist / self.interpolation_threshold)
                
                if num_points_to_insert > 0:
                    # 线性插值
                    lin_interp = interp1d([0, 1], np.vstack([points[start_idx], points[end_idx]]), 
                                         axis=0, kind='linear')
                    new_points = lin_interp(np.linspace(0, 1, num_points_to_insert + 2)[1:-1])
                    
                    # 添加插值点
                    interpolated.extend(new_points)
            
            interpolated.append(points[i])
        
        return np.array(interpolated)

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
        """提取凹边界"""
        polygon = alphashape.alphashape(points, self.alpha)
        if polygon and polygon.geom_type == 'Polygon':
            simplified = polygon.simplify(self.simplify_tolerance, preserve_topology=True)
            return np.array(simplified.exterior.coords)
        return []

    def _visualize(self, original_points, interpolated_points, polygons):
        """可视化结果"""
        plt.figure(figsize=(12, 8))
        
        # 绘制原始点和插值点
        plt.scatter(original_points[:, 0], original_points[:, 1], 
                   c='red', s=30, alpha=0.7, label='原始点', marker='o')
        plt.scatter(interpolated_points[:, 0], interpolated_points[:, 1], 
                   c='gray', s=5, alpha=0.3, label='插值点')
        
        # 绘制多边形区域
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
        
        plt.title('优化后的工作区域识别结果', fontsize=14)
        plt.xlabel('投影坐标 X (米)', fontsize=10)
        plt.ylabel('投影坐标 Y (米)', fontsize=10)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from getData import getData
    test_points = getData("data/2134.json")

    # 参数说明：
    # interpolation_threshold - 当点间距超过此值(米)时进行插值
    calculator = OptimizedAreaCalculator(density_radius=5, min_points=10, 
                                       alpha=0.3, interpolation_threshold=10)
    areas, total = calculator.calculate_work_areas(test_points, visualize=True)

    print(f"识别到 {len(areas)} 个工作区域")
    print(f"各区域面积（平方米）: {areas}")
    print(f"总面积: {total:.2f} 平方米 ({total / 666.67:.2f} 亩)")
