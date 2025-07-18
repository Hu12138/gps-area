from flask import Flask, request, jsonify
from pyproj import Transformer
from shapely.geometry import Polygon
import numpy as np
from datetime import datetime
from sklearn.cluster import DBSCAN
import hdbscan
import alphashape
import json

app = Flask(__name__)

class TrajectoryProcessor:
    def __init__(self):
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)

    def haversine(self, p1, p2):
        lon1, lat1, lon2, lat2 = map(np.radians, [p1["lon"], p1["lat"], p2["lon"], p2["lat"]])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 6371000 * 2 * np.arcsin(np.sqrt(a))

    def identify_linear_segments(self, points, speed_threshold=5, distance_threshold=2, angle_threshold=15, window=5):
        linear_segments = []
        current_segment = []

        def calculate_angle(p1, p2):
            dx, dy = p2["x"] - p1["x"], p2["y"] - p1["y"]
            return np.arctan2(dy, dx) * 180 / np.pi

        for p in points:
            p["x"], p["y"] = self.transformer.transform(p["lon"], p["lat"])

        for i in range(1, len(points)):
            p0, p1 = points[i - 1], points[i]
            speed = (p0["speed"] + p1["speed"]) / 2
            dist = self.haversine(p0, p1)
            angle = calculate_angle(p0, p1)

            if speed > speed_threshold and dist > distance_threshold:
                if current_segment and abs(calculate_angle(current_segment[-1], p1) - angle) > angle_threshold:
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

        if len(current_segment) >= window:
            linear_segment_indices = set(id(p) for p in current_segment)
            linear_segments.append(linear_segment_indices)

        return linear_segments

    def filter_linear_segments(self, points):
        linear_segments = self.identify_linear_segments(points)
        highway_points = set()
        for seg in linear_segments:
            highway_points.update(seg)
        filtered_points = []
        for p in points:
            if id(p) not in highway_points:
                filtered_points.append([p["lon"], p["lat"]])
        return np.array(filtered_points)

    def calculate_work_areas(self, points, group_min_size=50, group_cluster_size=30,
                              density_radius=5, min_points=13,
                              alpha=0.3, simplify_tolerance=2,
                              offset_distance=2):
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points])
        groups = self._auto_spatial_grouping(proj_points, group_min_size, group_cluster_size)
        all_polygons = []
        for group in groups:
            extended_points = self._add_direction_aware_points(group, offset_distance)
            clusters = self._density_clustering(extended_points, density_radius, min_points)
            for cluster in clusters:
                if len(cluster) < 3:
                    continue
                boundary = self._extract_alpha_shape(cluster, alpha, simplify_tolerance)
                if len(boundary) >= 3:
                    area = Polygon(boundary).area
                    if area > 10:
                        all_polygons.append(boundary)
        areas = [Polygon(poly).area for poly in all_polygons]
        total_area = sum(areas)
        return areas, total_area, all_polygons

    def _auto_spatial_grouping(self, points, min_size, cluster_size):
        if len(points) < min_size:
            return [points]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=cluster_size, cluster_selection_method='eom')
        labels = clusterer.fit_predict(points)
        groups = [points[labels == label] for label in set(labels) if label != -1]
        return groups if groups else [points]

    def _add_direction_aware_points(self, points, offset_distance):
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
        if len(points) < min_pts:
            return []
        db = DBSCAN(eps=radius, min_samples=min_pts).fit(points)
        return [points[db.labels_ == label] for label in set(db.labels_) if label != -1]

    def _extract_alpha_shape(self, points, alpha, tolerance):
        polygon = alphashape.alphashape(points, alpha)
        if polygon.geom_type == 'Polygon':
            simplified = polygon.simplify(tolerance, preserve_topology=True)
            return np.array(simplified.exterior.coords)
        return []

@app.route('/calculate_work_area', methods=['POST'])
def calculate_work_area():
    try:
        data = request.get_json()
        points = data.get("points", [])
        min_area = float(data.get("min_area", 0))
        tool_width = float(data.get("tool_width", 2))

        if not points:
            return jsonify({"status": "error", "message": "No points provided."}), 400

        parsed_points = []
        for p in points:
            parsed_points.append({
                "lon": float(p["lon"]),
                "lat": float(p["lat"]),
                "speed": float(p.get("speed", 0)),
                "time": p["time"]
            })

        processor = TrajectoryProcessor()
        filtered_points = processor.filter_linear_segments(parsed_points)

        areas, total_area, polygons = processor.calculate_work_areas(
            filtered_points,
            group_min_size=50,
            group_cluster_size=30,
            density_radius=5,
            min_points=13,
            alpha=0.3,
            offset_distance=tool_width
        )

        results = {}
        index = 0
        total_area_mu = 0
        for poly in polygons:
            polygon = Polygon(poly)
            area = polygon.area
            if area < min_area:
                continue
            centroid_x, centroid_y = polygon.centroid.x, polygon.centroid.y
            lon, lat = processor.transformer.transform(centroid_x, centroid_y, direction='INVERSE')
            results[f"field_{index}"] = {
                "area_m2": round(area, 2),
                "area_mu": round(area / 666.67, 2),
                "center_gps": [round(lon, 6), round(lat, 6)],
                "point_count": len(poly)
            }
            total_area_mu += area / 666.67
            index += 1

        return jsonify({
            "status": "success",
            "field_count": len(results),
            "fields": results,
            "total_area_m2": round(sum([Polygon(p).area for p in polygons if Polygon(p).area >= min_area]), 2),
            "total_area_mu": round(total_area_mu, 2)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)