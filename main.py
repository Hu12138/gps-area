from flask import Flask, request, jsonify
import numpy as np
from pyproj import Transformer
from datetime import datetime
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
import alphashape
import hdbscan

app = Flask(__name__)

class TrajectoryProcessor:
    def __init__(self):
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)
        self.transformer_back = Transformer.from_crs("EPSG:3410", "EPSG:4326", always_xy=True)
        
    def process_input_data(self, raw_data):
        """处理输入数据，确保类型正确"""
        points = raw_data.get("points", [])
        processed = []
        for p in points:
            try:
                processed.append({
                    "lon": float(p.get("lon", 0)),
                    "lat": float(p.get("lat", 0)),
                    "speed": float(p.get("speed", 0)),
                    "time": datetime.strptime(p["time"], "%Y-%m-%d %H:%M:%S") if "time" in p else datetime.now()
                })
            except (ValueError, KeyError) as e:
                continue  # 跳过无效数据点
        return processed
    
    def haversine(self, p1, p2):
        """安全的经纬度距离计算"""
        try:
            lon1, lat1, lon2, lat2 = map(np.radians, [p1["lon"], p1["lat"], p2["lon"], p2["lat"]])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            return 6371000 * 2 * np.arcsin(np.sqrt(a))
        except:
            return 0  # 如果计算失败返回0
    
    def filter_highway_points(self, points):
        """更安全的公路点过滤"""
        if len(points) < 2:
            return [[p["lon"], p["lat"]] for p in points]
            
        try:
            linear_segments = self._identify_linear_segments(points)
            highway_points = set().union(*linear_segments) if linear_segments else set()
            return [[p["lon"], p["lat"]] for p in points if id(p) not in highway_points]
        except:
            return [[p["lon"], p["lat"]] for p in points]  # 出错时返回所有点
    
    def _identify_linear_segments(self, points, speed_threshold=5, distance_threshold=2, angle_threshold=15, window=5):
        """更健壮的公路段识别"""
        if len(points) < 2:
            return []
            
        linear_segments = []
        current_segment = []

        # 坐标转换
        for p in points:
            try:
                p["x"], p["y"] = self.transformer.transform(p["lon"], p["lat"])
            except:
                p["x"], p["y"] = 0, 0  # 无效坐标设为0

        for i in range(1, len(points)):
            p0, p1 = points[i-1], points[i]
            
            # 安全计算
            speed = (float(p0.get("speed", 0)) + float(p1.get("speed", 0))) / 2
            dist = self.haversine(p0, p1)
            
            # 方向计算
            try:
                angle = np.arctan2(p1["y"]-p0["y"], p1["x"]-p0["x"]) * 180 / np.pi
            except:
                angle = 0

            if speed > speed_threshold and dist > distance_threshold:
                if len(current_segment) >= 2:  # 至少有2个点才能计算角度
                    try:
                        last_angle = np.arctan2(current_segment[-1]["y"]-current_segment[-2]["y"],
                                             current_segment[-1]["x"]-current_segment[-2]["x"]) * 180 / np.pi
                        if abs(last_angle - angle) > angle_threshold:
                            if len(current_segment) >= window:
                                linear_segments.append(set(id(p) for p in current_segment))
                            current_segment = [p1]
                            continue
                    except:
                        pass
                current_segment.append(p1)
            else:
                if len(current_segment) >= window:
                    linear_segments.append(set(id(p) for p in current_segment))
                current_segment = []

        if len(current_segment) >= window:
            linear_segments.append(set(id(p) for p in current_segment))
            
        return linear_segments
    
    def calculate_work_areas(self, filtered_points, min_area=200, **kwargs):
        """计算工作区域，增加最小面积过滤"""
        if not filtered_points:
            return [], 0
            
        try:
            proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in filtered_points])
            groups = self._auto_spatial_grouping(proj_points, 
                                              kwargs.get('group_min_size', 50), 
                                              kwargs.get('group_cluster_size', 30))
            
            all_polygons = []
            for group in groups:
                extended_points = self._add_direction_aware_points(group, kwargs.get('offset_distance', 2))
                clusters = self._density_clustering(extended_points, 
                                                 kwargs.get('density_radius', 5), 
                                                 kwargs.get('min_points', 15))
                
                for cluster in clusters:
                    if len(cluster) >= 3:
                        boundary = self._extract_alpha_shape(cluster, 
                                                          kwargs.get('alpha', 0.3), 
                                                          kwargs.get('simplify_tolerance', 2))
                        if len(boundary) >= 3:
                            poly = Polygon(boundary)
                            if poly.area > min_area:  # 使用传入的最小面积参数
                                all_polygons.append(boundary)
            
            results = []
            total_area = 0
            for poly in all_polygons:
                poly_obj = Polygon(poly)
                area = poly_obj.area
                try:
                    centroid = poly_obj.centroid
                    lon, lat = self.transformer_back.transform(centroid.x, centroid.y)
                    results.append({
                        "area_m2": area,
                        "point_count": len(poly),
                        "center_gps": [round(lon, 6), round(lat, 6)]
                    })
                    total_area += area
                except:
                    continue
                    
            return results, total_area
        except Exception as e:
            print(f"Error in calculate_work_areas: {str(e)}")
            return [], 0
    
    # 其余辅助方法保持不变...
    def _auto_spatial_grouping(self, points, min_size, cluster_size):
        """HDBSCAN空间分组"""
        if len(points) < min_size:
            return [points]
            
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_size,
                min_samples=cluster_size,
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(points)
            return [points[labels == label] for label in set(labels) if label != -1] or [points]
        except:
            return [points]

    def _add_direction_aware_points(self, points, offset_distance):
        """轨迹点插值"""
        if len(points) < 2:
            return points
            
        extended = [points[0]]
        for i in range(1, len(points)):
            p0, p1 = points[i-1], points[i]
            extended.append(p1)
            
            try:
                dist = np.linalg.norm(p1 - p0)
                if dist > offset_distance * 1.5:
                    num = int(dist // offset_distance)
                    for j in range(1, num):
                        interp = p0 + (p1 - p0) * (j/num)
                        extended.append(interp)
            except:
                continue
        return np.array(extended)

    def _density_clustering(self, points, radius, min_pts):
        """DBSCAN聚类"""
        if len(points) < min_pts:
            return []
        try:
            db = DBSCAN(eps=radius, min_samples=min_pts).fit(points)
            return [points[db.labels_ == label] for label in set(db.labels_) if label != -1]
        except:
            return []

    def _extract_alpha_shape(self, points, alpha, tolerance):
        """Alpha形状边界提取"""
        if len(points) < 3:
            return []
            
        try:
            polygon = alphashape.alphashape(points, alpha)
            if polygon.geom_type == 'Polygon':
                simplified = polygon.simplify(tolerance, preserve_topology=True)
                return np.array(simplified.exterior.coords)
            return []
        except:
            return []

@app.route('/calculate_work_area', methods=['POST'])
def calculate_work_area():
    try:
        # 获取输入数据
        input_data = request.get_json()
        if not input_data or "points" not in input_data:
            return jsonify({"status": "error", "message": "Invalid input format"}), 400
        
        # 初始化处理器
        processor = TrajectoryProcessor()
        
        # 处理输入数据
        points = processor.process_input_data(input_data)
        if not points:
            return jsonify({"status": "error", "message": "No valid points found"}), 400
            
        filtered_points = processor.filter_highway_points(points)
        
        # 获取参数
        min_area = float(input_data.get("min_area", 200))
        tool_width = float(input_data.get("tool_width", 2))
        
        # 计算工作区域
        params = {
            'group_min_size': max(10, int(len(points)/20)),  # 动态调整
            'group_cluster_size': max(5, int(len(points)/50)),
            'density_radius': tool_width * 2.5,
            'min_points': 15,
            'alpha': 0.3,
            'simplify_tolerance': tool_width,
            'offset_distance': tool_width/2,
            'min_area': min_area
        }
        
        field_data, total_area = processor.calculate_work_areas(filtered_points, **params)
        
        # 构建响应
        response = {
            "status": "success",
            "field_count": len(field_data),
            "total_area_m2": round(total_area, 2),
            "total_area_mu": round(total_area / 666.67, 2),
            "fields": {
                f"field_{i}": {
                    "area_m2": round(field["area_m2"], 2),
                    "area_mu": round(field["area_m2"] / 666.67, 2),
                    "point_count": field["point_count"],
                    "center_gps": field["center_gps"]
                }
                for i, field in enumerate(field_data)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Processing error: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)