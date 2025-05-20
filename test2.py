import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from pyproj import Transformer
from sklearn.cluster import DBSCAN
import alphashape  # pip install alphashape
import matplotlib.pyplot as plt
from collections import defaultdict

class FarmWorkAnalyzer:
    def __init__(self, tool_width=2.0, min_points=10):
        self.tool_width = tool_width
        self.min_points = min_points
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def analyze_work_areas(self, points, debug=False):
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points])
        avg_dist = self._calc_average_distance(proj_points)
        eps = max(self.tool_width * 2, avg_dist * 2)
        min_samples = max(5, int(len(points)*0.02))

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(proj_points)
        labels = db.labels_

        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:
                clusters[label].append(proj_points[idx])

        results = {}
        polygons = []
        for label, cluster in clusters.items():
            if len(cluster) >= self.min_points:
                poly = self._build_concave_hull(cluster)
                if poly is not None:
                    area = poly.area
                    results[label] = area
                    polygons.append((label, poly))
                    if debug:
                        print(f"区域{label}: {area:.2f} m²，{area/666.67:.2f}亩")

        total = sum(results.values())

        if debug:
            print(f"参数: eps={eps:.1f}m, min_samples={min_samples}")
            print(f"总面积: {total:.2f} m² ({total/666.67:.2f} 亩)")
            self._plot_results(proj_points, labels, polygons)

        return results, total

    def _calc_average_distance(self, points):
        if len(points) < 2:
            return self.tool_width * 2
        dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        return np.median(dists)

    def _build_concave_hull(self, points, alpha=None):
        try:
            if not alpha:
                alpha = 0.01  # 初始小 alpha，控制包络精度
            shape = alphashape.alphashape(points, alpha)
            if shape.is_valid and isinstance(shape, Polygon):
                return shape
        except Exception:
            return None
        return None

    def _plot_results(self, points, labels, polygons):
        plt.figure(figsize=(10, 6))
        unique_labels = set(labels)
        colors = [plt.cm.tab10(i) for i in range(len(unique_labels))]

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(points[mask, 0], points[mask, 1], c=[color], s=5, label=f'区域{label}' if label != -1 else '噪声')

        for label, poly in polygons:
            x, y = poly.exterior.xy
            plt.plot(x, y, color='black', linewidth=1.5)

        plt.legend()
        plt.title("工作区域识别与面积计算")
        plt.show()
def generate_test_data():
    """
    生成两块蛇形田+中间的转场路径（模拟农机在两个田之间移动）
    """
    def snake_track(start_lon, start_lat, length=100, width=50, rows=10):
        """
        模拟蛇形行走的轨迹，模拟农机在一个田块内的作业轨迹。
        """
        lon_scale = 0.00001 * length
        lat_scale = 0.00001 * width
        points = []
        for i in range(rows + 1):
            lat = start_lat + i * (lat_scale / rows)
            if i % 2 == 0:
                lons = np.linspace(start_lon, start_lon + lon_scale, 20)
            else:
                lons = np.linspace(start_lon + lon_scale, start_lon, 20)
            points.extend(zip(lons, [lat] * 20))
        return points

    # 两个田块，略微偏移
    track1 = snake_track(116.40, 39.90)
    track2 = snake_track(116.407, 39.904)

    # 中间的非作业路径（比如路上）
    transfer = [(116.40 + 0.0003 * i, 39.90 + 0.0002 * i) for i in range(20)]

    return track1 + transfer + track2

def generate_test_data_with_extremes():
    def snake_track(start_lon, start_lat, length=100, width=50, rows=10, points_per_row=20):
        lon_scale = 0.00001 * length
        lat_scale = 0.00001 * width
        points = []
        for i in range(rows + 1):
            lat = start_lat + i * (lat_scale / rows)
            if i % 2 == 0:
                lons = np.linspace(start_lon, start_lon + lon_scale, points_per_row)
            else:
                lons = np.linspace(start_lon + lon_scale, start_lon, points_per_row)
            points.extend(zip(lons, [lat] * points_per_row))
        return points

    # 场景1：极小田块
    small_field = snake_track(116.390, 39.890, length=10, width=5, rows=2, points_per_row=5)

    # 场景2：直线路径（非耕作移动）
    straight_line = [(116.395 + 0.0001 * i, 39.895 + 0.00005 * i) for i in range(30)]

    # 场景3：两个相邻但不连通的正常田块
    field1 = snake_track(116.400, 39.900)
    field2 = snake_track(116.407, 39.905)

    # 场景4：重复覆盖区域
    repeated = snake_track(116.410, 39.910) + snake_track(116.410, 39.910)

    # 场景5：轨迹断裂（模拟 GPS 丢失）
    broken_track = snake_track(116.420, 39.920, rows=5)
    broken_track = broken_track[:30] + broken_track[50:]  # 删掉中间20个点

    # 合并所有轨迹
    return small_field + straight_line + field1 + field2 + repeated + broken_track
if __name__ == "__main__":
    # test_data = generate_test_data()
    test_data = generate_test_data_with_extremes()
    analyzer = FarmWorkAnalyzer(tool_width=2.0)
    results, total = analyzer.analyze_work_areas(test_data, debug=True)