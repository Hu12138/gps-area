import json
from math import atan2, degrees, hypot
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from pyproj import Transformer


matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_data(path):
    from getData import getData
    raw = getData(path)
    # 转换为 (纬度, 经度)
    return [(lat, lon) for lon, lat in raw if isinstance(lat, float) and isinstance(lon, float)]


def calculate_angle(a, b, c):
    ba_x, ba_y = a[0] - b[0], a[1] - b[1]
    bc_x, bc_y = c[0] - b[0], c[1] - b[1]
    angle = degrees(atan2(ba_x * bc_y - ba_y * bc_x, ba_x * bc_x + ba_y * bc_y))
    return abs(angle)


def find_turning_zones(points, angle_threshold=20, window_size=3):
    zones = []
    current = []
    for i in range(1, len(points) - 1):
        angle = calculate_angle(points[i - 1], points[i], points[i + 1])
        if abs(180 - angle) > angle_threshold:
            current.append(i)
        else:
            if len(current) >= window_size:
                zones.append((current[0], current[-1]))
            current = []
    if len(current) >= window_size:
        zones.append((current[0], current[-1]))
    return zones


def extract_work_segments(points, turning_zones):
    if not turning_zones:
        return []
    boundaries = [0] + [z[1] for z in turning_zones] + [len(points) - 1]
    boundaries = sorted(set(boundaries))
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start >= 3:
            segments.append(points[start:end + 1])
    return segments


def segment_direction(segment):
    if len(segment) < 2:
        return None
    dx = segment[-1][1] - segment[0][1]
    dy = segment[-1][0] - segment[0][0]
    return degrees(atan2(dy, dx))


def group_similar_direction_segments(segments, angle_diff_thresh=25):
    """
    将连续角度相近的作业段分为同一田块
    """
    grouped = []
    current_group = []
    last_dir = None

    for seg in segments:
        angle = segment_direction(seg)
        if angle is None:
            continue
        if not current_group:
            current_group.append(seg)
            last_dir = angle
        else:
            if abs(angle - last_dir) < angle_diff_thresh:
                current_group.append(seg)
                last_dir = angle
            else:
                grouped.append(current_group)
                current_group = [seg]
                last_dir = angle

    if current_group:
        grouped.append(current_group)
    return grouped


def visualize(all_points, grouped_segments):
    plt.figure(figsize=(10, 8))
    lons = [p[1] for p in all_points]
    lats = [p[0] for p in all_points]
    plt.plot(lons, lats, color='gray', linewidth=0.5, label='原始轨迹')

    colors = plt.cm.get_cmap('tab10', 10)
    for i, group in enumerate(grouped_segments):
        for seg in group:
            lat = [p[0] for p in seg]
            lon = [p[1] for p in seg]
            plt.plot(lon, lat, linewidth=2, color=colors(i), label=f"田块组 {i}" if seg == group[0] else "")

    plt.title("基于规则识别的农机作业田块")
    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from getData import getData
    test_points = getData("data/2134.json")
    test_points = [[float(i[0]),float(i[1])] for i in test_points]
    print(f"✅ 原始点数：{len(test_points)}")

    turning_zones = find_turning_zones(test_points)
    print(f"✅ 掉头区域数：{len(turning_zones)}")

    work_segments = extract_work_segments(test_points, turning_zones)
    print(f"✅ 作业段数：{len(work_segments)}")

    valid_segments = []
    for seg in work_segments:
        if len(seg) < 2:
            continue
        angle = segment_direction(seg)
        if angle is None:
            continue
        length = hypot(seg[-1][1] - seg[0][1], seg[-1][0] - seg[0][0])
        if length > 0.0001:
            valid_segments.append(seg)

    grouped = group_similar_direction_segments(valid_segments)
    print(f"✅ 有效田块组数量: {len(grouped)}")

    visualize(test_points, grouped)