import numpy as np
from shapely.geometry import Polygon
from scipy.interpolate import interp1d
from pyproj import Transformer
from datetime import datetime, timedelta


class OptimizedAreaCalculator:
    def __init__(self, epsg_code="32650", interpolation_threshold=10):
        """
        epsg_code: 要投影的坐标系（默认是 UTM 50N）
        interpolation_threshold: 插值点之间的最大距离（单位：米）
        """
        self.transformer = Transformer.from_crs("epsg:4326", f"epsg:{epsg_code}", always_xy=True)
        self.interpolation_threshold = interpolation_threshold

    def _adaptive_interpolation(self, points, timestamps=None):
        """支持时间戳的自适应线性插值"""
        if len(points) < 2:
            return points

        if timestamps:
            # 转换字符串时间为 datetime 对象
            timestamps = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in timestamps]
        else:
            timestamps = [None] * len(points)

        interpolated = [points[0]]
        interpolated_times = [timestamps[0]]

        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i - 1])
            time_gap = (timestamps[i] - timestamps[i - 1]).total_seconds() if timestamps[i] and timestamps[i - 1] else 0

            # 判断是否需要插值（距离阈值或时间阈值）
            needs_interp = dist > self.interpolation_threshold or time_gap > 2  # 时间阈值可调

            if needs_interp:
                num_points = max(
                    int(dist / self.interpolation_threshold),
                    int(time_gap / 2) if time_gap else 0
                )
                num_points = max(1, num_points)

                # 空间插值
                lin_interp = interp1d([0, 1], np.vstack([points[i - 1], points[i]]), axis=0)
                new_points = lin_interp(np.linspace(0, 1, num_points + 2)[1:-1])
                interpolated.extend(new_points)

                # 时间插值
                if timestamps[i - 1] and timestamps[i]:
                    delta = (timestamps[i] - timestamps[i - 1]) / (num_points + 1)
                    new_times = [timestamps[i - 1] + delta * j for j in range(1, num_points + 1)]
                    interpolated_times.extend(new_times)
                else:
                    interpolated_times.extend([None] * num_points)

            interpolated.append(points[i])
            interpolated_times.append(timestamps[i])

        return np.array(interpolated), interpolated_times

    def calculate_area(self, gps_points, timestamps=None):
        """
        gps_points: List of (lon, lat) tuples
        timestamps: List of timestamp strings ("YYYY-MM-DD HH:MM:SS")，可选
        """
        if len(gps_points) < 3:
            return 0

        # 坐标转换：WGS84 => 投影坐标系
        projected_points = np.array([self.transformer.transform(lon, lat) for lon, lat in gps_points])

        # 插值
        interpolated, _ = self._adaptive_interpolation(projected_points, timestamps)

        # 构建多边形并计算面积
        polygon = Polygon(interpolated)
        return polygon.area
    

if __name__ == "__main__":
    from getData import getData
    test_points = getData("data/gps-time.json")

    # 参数说明：
    # interpolation_threshold - 当点间距超过此值(米)时进行插值
    calculator = OptimizedAreaCalculator(density_radius=5, min_points=10, 
                                       alpha=0.3, interpolation_threshold=10)
    areas, total = calculator.calculate_work_areas(test_points, visualize=True)

    print(f"识别到 {len(areas)} 个工作区域")
    print(f"各区域面积（平方米）: {areas}")
    print(f"总面积: {total:.2f} 平方米 ({total / 666.67:.2f} 亩)")