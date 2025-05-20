from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
from pyproj import Transformer
import alphashape

app = Flask(__name__)

class SimpleAreaCalculator:
    def __init__(self, density_radius=5, min_points=20, alpha=0.05, simplify_tolerance=2.0):
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)
        self.density_radius = density_radius
        self.min_points = min_points
        self.alpha = alpha
        self.simplify_tolerance = simplify_tolerance

    def calculate_work_areas(self, points):
        proj_points = np.array([self.transformer.transform(float(lon), float(lat)) for lon, lat in points])
        clusters = self._density_clustering(proj_points)

        polygons = []
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            boundary_points = self._extract_alpha_shape(cluster)
            if len(boundary_points) >= 3:
                polygons.append(boundary_points)

        areas = []
        valid_polygons = []
        for poly in polygons:
            area = Polygon(poly).area
            if area > 100:
                areas.append((area, len(poly)))
                valid_polygons.append(poly)

        return areas, sum(a[0] for a in areas)

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
        areas, total_area = calculator.calculate_work_areas(points)

        fields = {}
        for i, (area, point_count) in enumerate(areas):
            fields[f'field_{i}'] = {
                'area_m2': round(area, 2),
                'area_mu': round(area / 666.67, 2),
                'point_count': point_count
            }

        response = {
            "status": "success",
            "field_count": len(areas),
            "total_area_m2": round(total_area, 2),
            "total_area_mu": round(total_area / 666.67, 2),
            "fields": fields
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)