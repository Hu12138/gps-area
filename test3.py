import numpy as np
from shapely.geometry import LineString
from pyproj import Transformer
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

# ✅ 设置支持中文字体（适用于 macOS 和常见平台）
matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class FarmWorkAnalyzer:
    def __init__(self, tool_width=2.0, min_points=10, min_area=100):
        self.tool_width = tool_width
        self.min_points = min_points
        self.min_area = min_area
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def analyze_work_areas(self, points, debug=False):
        proj_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points])

        eps = self.tool_width * 1.5
        min_samples = max(2, int(len(points) * 0.005))

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(proj_points)

        clusters = defaultdict(list)
        for i, label in enumerate(db.labels_):
            if label != -1:
                clusters[label].append(proj_points[i])

        results = {}
        if debug:
            print(f"[DEBUG] eps: {eps:.2f}, min_samples: {min_samples}, 总点数: {len(points)}, 聚类数: {len(clusters)}")

        for label, cluster in clusters.items():
            if len(cluster) >= self.min_points:
                if self._is_straight_path(cluster):
                    if debug:
                        print(f"区域{label} 忽略：疑似直线行驶路径")
                    continue
                line = LineString(cluster)
                area = line.buffer(self.tool_width / 2).area
                if area >= self.min_area:
                    results[label] = area
                    if debug:
                        print(f"区域{label}: {area:.2f} m²，{area/666.67:.2f}亩 (点数: {len(cluster)})")
                else:
                    if debug:
                        print(f"区域{label} 忽略：面积过小 ({area:.2f} m²)")
            else:
                if debug:
                    print(f"区域{label} 忽略：点数不足 ({len(cluster)} 点)")

        total = sum(results.values())
        if debug:
            print(f"参数: eps={eps:.1f}m, min_samples={min_samples}")
            print(f"总面积: {total:.2f} m² ({total/666.67:.2f} 亩)")
            self._plot_results(proj_points, db.labels_)
        return results, total

    def _is_straight_path(self, cluster, angle_threshold=5, max_variance=10):
        if len(cluster) < 3:
            return True
        angles = []
        for i in range(1, len(cluster) - 1):
            v1 = np.array(cluster[i]) - np.array(cluster[i - 1])
            v2 = np.array(cluster[i + 1]) - np.array(cluster[i])
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            angles.append(angle)
        if not angles:
            return True
        return np.std(angles) < max_variance

    def _plot_results(self, points, labels):
        plt.figure(figsize=(10, 6))
        unique_labels = set(labels)
        colors = [plt.cm.tab10(i % 10) for i in unique_labels]

        for label, color in zip(unique_labels, colors):
            mask = (labels == label)
            if label == -1:
                plt.scatter(points[mask, 0], points[mask, 1],
                            c='gray', alpha=0.3, s=5, label='噪声')
            else:
                plt.scatter(points[mask, 0], points[mask, 1],
                            c=[color], s=8, label=f'区域{label}')
        plt.legend()
        plt.title("工作区域识别结果")
        plt.xlabel("投影 X")
        plt.ylabel("投影 Y")
        plt.tight_layout()
        plt.show()

def generate_extreme_test_data():
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

    small_field = snake_track(116.390, 39.890, length=10, width=5, rows=2, points_per_row=5)
    straight_line = [(116.395 + 0.0001 * i, 39.895 + 0.00005 * i) for i in range(30)]
    field1 = snake_track(116.400, 39.900)
    field2 = snake_track(116.407, 39.905)
    repeated = snake_track(116.410, 39.910) + snake_track(116.410, 39.910)
    broken_track = snake_track(116.420, 39.920, rows=5)
    broken_track = broken_track[:30] + broken_track[50:]

    return small_field + straight_line + field1 + field2 + repeated + broken_track

if __name__ == "__main__":
    # test_data = generate_extreme_test_data()
    from getData import getData
    test_data = getData("data/test1.json")
    analyzer = FarmWorkAnalyzer(tool_width=4.0)
    results, total = analyzer.analyze_work_areas(test_data, debug=True)