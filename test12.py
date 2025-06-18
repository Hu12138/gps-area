import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
import alphashape
import matplotlib.pyplot as plt
import matplotlib
from pyproj import Transformer

matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class OptimizedAreaCalculator:
    def __init__(self, density_radius=5, min_points=20, alpha=0.05,
                 simplify_tolerance=2.0, offset_distance=3,
                 extend_main_direction=True, extend_other_directions=True):
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)
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

    def calculate_work_areas(self, points, visualize=False):
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points])
        extended_points = self._add_direction_aware_points(proj_points)
        clusters = self._density_clustering(extended_points)

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
            if area > 10:
                areas.append(area)
                valid_polygons.append(poly)

        if visualize:
            self._visualize(proj_points, extended_points, valid_polygons)

        return areas, sum(areas)

    def _get_dominant_direction(self, vec):
        norm_vec = vec / (np.linalg.norm(vec) + 1e-6)
        similarities = self.norm_directions @ norm_vec
        idx = np.argmax(similarities)
        return self.norm_directions[idx], similarities[idx]

    def _add_direction_aware_points(self, points, angle_cos_threshold=0.96, max_gap=3.0):
        extended = []
        extended.append(points[0])  # 首点保留

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

            # 如果方向一致，且中间距离太大，插值
            if cos_angle > angle_cos_threshold:
                total_dist = np.linalg.norm(p2 - p0)
                if total_dist > max_gap:
                    num_extra = int(total_dist // self.offset_distance)
                    direction = (p2 - p0) / total_dist
                    for j in range(1, num_extra):
                        interp_point = p0 + direction * j * self.offset_distance
                        extended.append(interp_point)

        extended.append(points[-1])  # 末点保留
        return np.array(extended)
    
    def _density_clustering(self, points):
        db = DBSCAN(eps=self.density_radius, min_samples=self.min_points).fit(points)
        clusters = []
        for label in set(db.labels_):
            if label == -1:
                continue
            cluster = points[db.labels_ == label]
            clusters.append(cluster)
        return clusters

    def _extract_alpha_shape(self, points):
        polygon = alphashape.alphashape(points, self.alpha)
        if polygon and polygon.geom_type == 'Polygon':
            simplified = polygon.simplify(self.simplify_tolerance, preserve_topology=True)
            return np.array(simplified.exterior.coords)
        return []

    def _visualize(self, original_points, extended_points, polygons):
        plt.figure(figsize=(12, 8))

        plt.scatter(original_points[:, 0], original_points[:, 1],
                    c='red', s=30, alpha=0.8, label='原始点')
        plt.scatter(extended_points[:, 0], extended_points[:, 1],
                    c='gray', s=5, alpha=0.4, label='扩展点')

        for i, poly in enumerate(polygons):
            polygon = Polygon(poly)
            x, y = polygon.exterior.xy
            plt.plot(x, y, linewidth=2, label=f'区域{i + 1}')
            plt.fill(x, y, alpha=0.2)

            centroid = polygon.centroid
            plt.text(centroid.x, centroid.y,
                     f'{polygon.area:.0f}㎡\n({polygon.area / 666.67:.1f}亩)',
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.8))

        plt.title('工作区域识别（方向感知扩展）')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True, alpha=0.3)

        # ⭐ 图例放到图外
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
    
if __name__ == "__main__":
    from getData import getData

    test_points = getData("data/test1.json")
    # test_points = getData("data/13885004840-11.json")
    # test_points = getData("data/13800002122-14.json")
    # test_points = getData("data/2134.json")

    calculator = OptimizedAreaCalculator(
        density_radius=5,
        min_points=15,
        alpha=0.3,
        simplify_tolerance=2,
        offset_distance=2,
        extend_main_direction=True,
        extend_other_directions=False  # 🚨只启用主方向扩展，关闭其他方向插值
    )

    areas, total = calculator.calculate_work_areas(test_points, visualize=True)

    print(f"识别到 {len(areas)} 个工作区域")
    print(f"各区域面积（平方米）: {areas}")
    print(f"总面积: {total:.2f} 平方米 ({total / 666.67:.2f} 亩)")