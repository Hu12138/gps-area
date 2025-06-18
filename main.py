from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
from pyproj import Transformer
import alphashape

app = Flask(__name__)


class OptimizedAreaCalculator:
    def __init__(self, density_radius=5, min_points=20, alpha=0.05,
                 simplify_tolerance=2.0, offset_distance=3,
                 extend_main_direction=True, extend_other_directions=True):
        self.transformer_to_proj = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)
        self.transformer_to_gps = Transformer.from_crs("EPSG:3410", "EPSG:4326", always_xy=True)
        self.density_radius = density_radius
        self.min_points = min_points
        self.alpha = alpha
        self.simplify_tolerance = simplify_tolerance
        self.offset_distance = offset_distance

        self.extend_main_direction = extend_main_direction
        self.extend_other_directions = extend_other_directions

        self.base_directions = np.array([
            [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, -1], [1, -1], [-1, 1]
        ])
        self.norm_directions = self.base_directions / np.linalg.norm(self.base_directions, axis=1, keepdims=True)

    def calculate_work_areas(self, points):
        proj_points = np.array([self.transformer_to_proj.transform(lon, lat) for lon, lat in points])
        extended_points = self._add_direction_aware_points(proj_points)
        clusters = self._density_clustering(extended_points)

        polygons = []
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            boundary_points = self._extract_alpha_shape(cluster)
            if len(boundary_points) >= 3:
                polygons.append(boundary_points)

        results = []
        total_area = 0
        for poly in polygons:
            shapely_poly = Polygon(poly)
            area = shapely_poly.area
            if area > 100:
                centroid = shapely_poly.centroid
                lon, lat = self.transformer_to_gps.transform(centroid.x, centroid.y)
                results.append({
                    "area_m2": area,
                    "point_count": len(poly),
                    "center_gps": [round(lon, 6), round(lat, 6)]
                })
                total_area += area

        return results, total_area

    def _add_direction_aware_points(self, points, angle_cos_threshold=0.96, max_gap=3.0):
        extended = [points[0]]
        for i in range(1, len(points) - 1):
            p0, p1, p2 = points[i - 1], points[i], points[i + 1]
            v1 = p1 - p0
            v2 = p2 - p1
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 < 1e-6 or norm2 < 1e-6:
                extended.append(p1)
                continue

            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            extended.append(p1)

            if cos_angle > angle_cos_threshold:
                total_dist = np.linalg.norm(p2 - p0)
                if total_dist > max_gap:
                    num_extra = int(total_dist // self.offset_distance)
                    direction = (p2 - p0) / total_dist
                    for j in range(1, num_extra):
                        interp_point = p0 + direction * j * self.offset_distance
                        extended.append(interp_point)

        extended.append(points[-1])
        return np.array(extended)

    def _density_clustering(self, points):
        db = DBSCAN(eps=self.density_radius, min_samples=self.min_points).fit(points)
        clusters = []
        for label in set(db.labels_):
            if label == -1:
                continue
            cluster = points[db.labels_ == label]
            if len(cluster) >= 3:
                clusters.append(cluster)
        return clusters

    def _extract_alpha_shape(self, points):
        polygon = alphashape.alphashape(points, self.alpha)
        if polygon and polygon.geom_type == 'Polygon':
            simplified = polygon.simplify(self.simplify_tolerance, preserve_topology=True)
            return np.array(simplified.exterior.coords)
        return []


@app.route('/calculate_plowing_area', methods=['POST'])
def calculate_area():
    data = request.get_json()
    if not data or 'points' not in data:
        return jsonify({"status": "error", "message": "Missing 'points' in request"}), 400

    try:
        points = data['points']  # list of [lon, lat]
        calculator = OptimizedAreaCalculator(
            density_radius=5,
            min_points=15,
            alpha=0.3,
            simplify_tolerance=2,
            offset_distance=2,
            extend_main_direction=True,
            extend_other_directions=False  # 可以改为 True
        )
        field_data, total_area = calculator.calculate_work_areas(points)

        fields = {}
        for i, field in enumerate(field_data):
            fields[f'field_{i}'] = {
                'area_m2': round(field["area_m2"], 2),
                'area_mu': round(field["area_m2"] / 666.67, 2),
                'point_count': field["point_count"],
                'center_gps': field["center_gps"]
            }

        response = {
            "status": "success",
            "field_count": len(field_data),
            "total_area_m2": round(total_area, 2),
            "total_area_mu": round(total_area / 666.67, 2),
            "fields": fields
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)