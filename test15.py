import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
import alphashape
import matplotlib.pyplot as plt
import matplotlib
from pyproj import Transformer
import hdbscan
matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class AutoSpatialGrouper:
    def __init__(self, 
                 group_min_size=50,      # HDBSCAN最小簇大小
                 group_cluster_size=30,  # HDBSCAN簇大小参数
                 density_radius=5,       # DBSCAN半径
                 min_points=15,          # DBSCAN最小点数
                 alpha=0.3,              # Alpha shape参数
                 simplify_tolerance=2,   # 简化阈值
                 offset_distance=2):     # 插值距离
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)
        self.group_min_size = group_min_size
        self.group_cluster_size = group_cluster_size
        self.density_radius = density_radius
        self.min_points = min_points
        self.alpha = alpha
        self.simplify_tolerance = simplify_tolerance
        self.offset_distance = offset_distance

    def calculate_work_areas(self, points, visualize=False):
        print("=== 开始处理 ===")
        print(f"原始数据点数量: {len(points)}")
        
        # 坐标转换
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points])
        print("坐标转换完成")
        
        # 自动空间分组
        print("\n进行自动空间分组...")
        groups = self._auto_spatial_grouping(proj_points)
        print(f"自动分成 {len(groups)} 个空间组")
        for i, group in enumerate(groups, 1):
            print(f"  第{i}组: {len(group)} 个点")
        
        # 处理每组数据
        all_polygons = []
        for i, group in enumerate(groups, 1):
            print(f"\n处理第{i}组 ({len(group)}点)...")
            
            # 插值增强
            extended_points = self._add_direction_aware_points(group)
            print(f"  插值后: {len(extended_points)}点")
            
            # 密度聚类
            clusters = self._density_clustering(extended_points)
            print(f"  找到子聚类: {len(clusters)}个")
            
            # 生成多边形
            for j, cluster in enumerate(clusters, 1):
                if len(cluster) < 3:
                    continue
                boundary = self._extract_alpha_shape(cluster)
                if len(boundary) >= 3:
                    area = Polygon(boundary).area
                    if area > 10:  # 过滤小区域
                        all_polygons.append(boundary)
                        print(f"    子聚类{j}面积: {area:.1f}㎡")
        
        # 计算总面积
        areas = [Polygon(poly).area for poly in all_polygons]
        total_area = sum(areas)
        
        print("\n=== 最终结果 ===")
        print(f"识别到 {len(areas)} 个工作区域")
        print(f"总面积: {total_area:.2f} 平方米 ({total_area / 666.67:.2f} 亩)")

        if visualize:
            self._visualize(proj_points, groups, all_polygons)
            
        return areas, total_area

    def _auto_spatial_grouping(self, points):
        """使用HDBSCAN自动空间分组"""
        if len(points) < self.group_min_size:
            return [points]
            
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.group_min_size,
            min_samples=self.group_cluster_size,
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(points)
        
        # 分组并过滤噪声点（label=-1）
        groups = []
        for label in set(labels):
            if label == -1:
                continue
            group = points[labels == label]
            groups.append(group)
            
        return groups if groups else [points]

    def _add_direction_aware_points(self, points):
        """带方向感知的插值"""
        if len(points) < 2:
            return points
            
        extended = [points[0]]
        for i in range(1, len(points)):
            p0, p1 = points[i-1], points[i]
            extended.append(p1)
            
            dist = np.linalg.norm(p1 - p0)
            if dist > self.offset_distance * 1.5:
                num = int(dist // self.offset_distance)
                for j in range(1, num):
                    interp = p0 + (p1 - p0) * (j/num)
                    extended.append(interp)
        return np.array(extended)

    def _density_clustering(self, points):
        """组内精细聚类"""
        if len(points) < self.min_points:
            return []
            
        db = DBSCAN(eps=self.density_radius, min_samples=self.min_points).fit(points)
        return [points[db.labels_ == label] for label in set(db.labels_) if label != -1]

    def _extract_alpha_shape(self, points):
        """提取多边形边界"""
        polygon = alphashape.alphashape(points, self.alpha)
        if polygon.geom_type == 'Polygon':
            simplified = polygon.simplify(self.simplify_tolerance, preserve_topology=True)
            return np.array(simplified.exterior.coords)
        return []

    def _visualize(self, all_points, groups, polygons):
        """增强可视化"""
        plt.figure(figsize=(15, 10))
        
        # 绘制所有点
        plt.scatter(all_points[:,0], all_points[:,1], 
                   c='gray', s=5, alpha=0.1, label='所有轨迹点')
        
        # 绘制分组结果
        colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
        for i, (group, color) in enumerate(zip(groups, colors), 1):
            plt.scatter(group[:,0], group[:,1], 
                       c=[color], s=20, alpha=0.6, label=f'空间组{i}')
        
        # 绘制多边形
        for i, poly in enumerate(polygons, 1):
            poly_obj = Polygon(poly)
            x, y = poly_obj.exterior.xy
            plt.plot(x, y, 'k-', linewidth=2, label=f'作业区{i}' if i==1 else "")
            plt.fill(x, y, alpha=0.2)
            
            centroid = poly_obj.centroid
            plt.text(centroid.x, centroid.y,
                    f'{poly_obj.area:.0f}㎡\n({poly_obj.area/666.67:.1f}亩)',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title('空间自动分组结果', fontsize=16)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    from getData import getData
    # test_points = getData("data/test1.json")
    # test_points = getData("data/13885004840-11.json")
    # test_points = getData("data/13800002122-14.json")
    # test_points = getData("data/13800002122-15 copy.json")
    # test_points = getData("data/13800002122-15.json")
    test_points = getData("data/2134.json")
    print(f"加载到 {len(test_points)} 个测试点")
    
    calculator = AutoSpatialGrouper(
        group_min_size=50,      # 空间组最小点数
        group_cluster_size=30,  # 空间组密度参数
        density_radius=5,       # 子聚类半径(米)
        min_points=13,          # 子聚类最小点数
        alpha=0.3,              # 边界紧密度
        offset_distance=2       # 插值间距(米)
    )
    
    areas, total = calculator.calculate_work_areas(test_points, visualize=True)