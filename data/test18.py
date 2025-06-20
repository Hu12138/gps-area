import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import json
from datetime import datetime
import matplotlib
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
import alphashape
import hdbscan

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class TrajectoryProcessor:
    def __init__(self):
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)
        
    def get_data(self, path):
        """加载原始数据"""
        with open(path, "r") as f:
            raw = json.load(f)
        raw = raw.get("points")
        return [
            {
                "lon": float(p["lon"]),
                "lat": float(p["lat"]),
                "speed": float(p.get("speed", 0)),
                "time": datetime.strptime(p["time"], "%Y-%m-%d %H:%M:%S")
            }
            for p in raw if "lon" in p and "lat" in p
        ]
    
    def haversine(self, p1, p2):
        """经纬度距离（米）"""
        lon1, lat1, lon2, lat2 = map(np.radians, [p1["lon"], p1["lat"], p2["lon"], p2["lat"]])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 6371000 * 2 * np.arcsin(np.sqrt(a))
    
    def identify_linear_segments(self, points, speed_threshold=5, distance_threshold=2, angle_threshold=15, window=5):
        """识别疑似公路轨迹段"""
        linear_segments = []
        current_segment = []

        def calculate_angle(p1, p2):
            dx, dy = p2["x"] - p1["x"], p2["y"] - p1["y"]
            return np.arctan2(dy, dx) * 180 / np.pi

        # 坐标转换
        for p in points:
            p["x"], p["y"] = self.transformer.transform(p["lon"], p["lat"])

        for i in range(1, len(points)):
            p0, p1 = points[i - 1], points[i]
            speed = (p0["speed"] + p1["speed"]) / 2
            dist = self.haversine(p0, p1)
            angle = calculate_angle(p0, p1)

            if speed > speed_threshold and dist > distance_threshold:
                if current_segment and abs(calculate_angle(current_segment[-1], p1) - angle) > angle_threshold:
                    # 方向突变，分段
                    if len(current_segment) >= window:
                        linear_segment_indices = set(id(p) for p in current_segment)
                        linear_segments.append(linear_segment_indices)
                    current_segment = [p1]
                else:
                    current_segment.append(p1)
            else:
                if len(current_segment) >= window:
                    linear_segment_indices = set(id(p) for p in current_segment)
                    linear_segments.append(linear_segment_indices)
                current_segment = []

        # 收尾
        if len(current_segment) >= window:
            linear_segment_indices = set(id(p) for p in current_segment)
            linear_segments.append(linear_segment_indices)

        return linear_segments
    
    def filter_linear_segments(self, points):
        """过滤掉疑似公路轨迹段"""
        linear_segments = self.identify_linear_segments(points)
        
        # 合并所有公路点的ID
        highway_points = set()
        for seg in linear_segments:
            highway_points.update(seg)
        
        # 过滤掉公路点
        filtered_points = []
        for p in points:
            if id(p) not in highway_points:
                filtered_points.append([p["lon"], p["lat"]])
        
        return np.array(filtered_points)
    
    def calculate_work_areas(self, points, visualize=False, 
                           group_min_size=50, group_cluster_size=30,
                           density_radius=5, min_points=15,
                           alpha=0.3, simplify_tolerance=2,
                           offset_distance=2):
        """计算工作区域"""
        print("=== 开始处理 ===")
        print(f"原始数据点数量: {len(points)}")
        
        # 坐标转换
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points])
        print("坐标转换完成")
        
        # 自动空间分组
        print("\n进行自动空间分组...")
        groups = self._auto_spatial_grouping(proj_points, group_min_size, group_cluster_size)
        print(f"自动分成 {len(groups)} 个空间组")
        for i, group in enumerate(groups, 1):
            print(f"  第{i}组: {len(group)} 个点")
        
        # 处理每组数据
        all_polygons = []
        for i, group in enumerate(groups, 1):
            print(f"\n处理第{i}组 ({len(group)}点)...")
            
            # 插值增强
            extended_points = self._add_direction_aware_points(group, offset_distance)
            print(f"  插值后: {len(extended_points)}点")
            
            # 密度聚类
            clusters = self._density_clustering(extended_points, density_radius, min_points)
            print(f"  找到子聚类: {len(clusters)}个")
            
            # 生成多边形
            for j, cluster in enumerate(clusters, 1):
                if len(cluster) < 3:
                    continue
                boundary = self._extract_alpha_shape(cluster, alpha, simplify_tolerance)
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
    
    def _auto_spatial_grouping(self, points, min_size, cluster_size):
        """使用HDBSCAN自动空间分组"""
        if len(points) < min_size:
            return [points]
            
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            min_samples=cluster_size,
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

    def _add_direction_aware_points(self, points, offset_distance):
        """带方向感知的插值"""
        if len(points) < 2:
            return points
            
        extended = [points[0]]
        for i in range(1, len(points)):
            p0, p1 = points[i-1], points[i]
            extended.append(p1)
            
            dist = np.linalg.norm(p1 - p0)
            if dist > offset_distance * 1.5:
                num = int(dist // offset_distance)
                for j in range(1, num):
                    interp = p0 + (p1 - p0) * (j/num)
                    extended.append(interp)
        return np.array(extended)

    def _density_clustering(self, points, radius, min_pts):
        """组内精细聚类"""
        if len(points) < min_pts:
            return []
            
        db = DBSCAN(eps=radius, min_samples=min_pts).fit(points)
        return [points[db.labels_ == label] for label in set(db.labels_) if label != -1]

    def _extract_alpha_shape(self, points, alpha, tolerance):
        """提取多边形边界"""
        polygon = alphashape.alphashape(points, alpha)
        if polygon.geom_type == 'Polygon':
            simplified = polygon.simplify(tolerance, preserve_topology=True)
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
    processor = TrajectoryProcessor()
    
    # 加载数据
    data = processor.get_data("data/13800002143-15-t-s.json")
    data = processor.get_data("data/13885004840-11-t-s.json")
    data = processor.get_data("data/13800002122-15-t-s.txt")
    print(f"加载到原始 {len(data)} 个测试点")
    
    # 过滤公路轨迹段
    filtered_points = processor.filter_linear_segments(data)
    print(f"过滤后剩余 {len(filtered_points)} 个点")
    
    # 计算工作区域
    areas, total = processor.calculate_work_areas(
        filtered_points, 
        visualize=True,
        group_min_size=50,
        group_cluster_size=30,
        density_radius=5,
        min_points=13,
        alpha=0.3,
        offset_distance=2
    )