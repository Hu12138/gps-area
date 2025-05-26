from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
from pyproj import Transformer
import alphashape

app = Flask(__name__)

class SimpleAreaCalculator:
    def __init__(self, density_radius=5, min_points=20, alpha=0.05, simplify_tolerance=2.0):
        self.transformer_to_proj = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)
        self.transformer_to_gps = Transformer.from_crs("EPSG:3410", "EPSG:4326", always_xy=True)
        self.density_radius = density_radius
        self.min_points = min_points
        self.alpha = alpha
        self.simplify_tolerance = simplify_tolerance

    def calculate_work_areas(self, points):
        proj_points = np.array([self.transformer_to_proj.transform(float(lon), float(lat)) for lon, lat in points])
        clusters = self._density_clustering(proj_points)

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
                lon, lat = self._centroid_to_gps(centroid.x, centroid.y)
                results.append({
                    "area_m2": area,
                    "point_count": len(poly),
                    "center_gps": [round(lon, 6), round(lat, 6)]
                })
                total_area += area

        return results, total_area

    def _centroid_to_gps(self, x, y):
        lon, lat = self.transformer_to_gps.transform(x, y)
        return lon, lat

    def _density_clustering(self, points):
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
        points = data['points']
        calculator = SimpleAreaCalculator(density_radius=5, min_points=10, alpha=0.3)
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
    app.run(debug=True)