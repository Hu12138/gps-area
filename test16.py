import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import json
from datetime import datetime
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)

def getData(path):
    with open(path, "r") as f:
        raw = json.load(f)
    raw = raw.get("points")
    return [
        {
            "lon": float(p["lon"]),
            "lat": float(p["lat"]),
            "speed": float(p.get("speed", 0)),
            "time": datetime.strptime(p["time"], "%Y-%m-%d %H:%M:%S")
        }
        for p in raw if "lon" in p and "lat" in p
    ]

def haversine(p1, p2):
    # 经纬度距离（米）
    lon1, lat1, lon2, lat2 = map(np.radians, [p1["lon"], p1["lat"], p2["lon"], p2["lat"]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))

def identify_linear_segments(points, speed_threshold=5, distance_threshold=2, angle_threshold=15, window=5):
    linear_segments = []
    current_segment = []

    def calculate_angle(p1, p2):
        dx, dy = p2["x"] - p1["x"], p2["y"] - p1["y"]
        return np.arctan2(dy, dx) * 180 / np.pi

    # 坐标转换
    for p in points:
        p["x"], p["y"] = transformer.transform(p["lon"], p["lat"])

    for i in range(1, len(points)):
        p0, p1 = points[i - 1], points[i]
        speed = (p0["speed"] + p1["speed"]) / 2
        dist = haversine(p0, p1)
        angle = calculate_angle(p0, p1)

        if speed > speed_threshold and dist > distance_threshold:
            if current_segment and abs(calculate_angle(current_segment[-1], p1) - angle) > angle_threshold:
                # 方向突变，分段
                if len(current_segment) >= window:
                    linear_segments.append(current_segment)
                current_segment = [p1]
            else:
                current_segment.append(p1)
        else:
            if len(current_segment) >= window:
                linear_segments.append(current_segment)
            current_segment = []

    # 收尾
    if len(current_segment) >= window:
        linear_segments.append(current_segment)

    return linear_segments

def plot_linear_segments(points, linear_segments):
    plt.figure(figsize=(12, 8))
    # 所有点
    all_x = [p["x"] for p in points]
    all_y = [p["y"] for p in points]
    plt.plot(all_x, all_y, c='gray', linewidth=1, alpha=0.3, label="全部轨迹")

    # 高亮公路轨迹段
    for i, seg in enumerate(linear_segments):
        seg_x = [p["x"] for p in seg]
        seg_y = [p["y"] for p in seg]
        plt.plot(seg_x, seg_y, linewidth=2.5, label=f"线性段{i+1}")

    plt.title("疑似公路轨迹段识别")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # data = getData("data/13800002143-15-t-s.json")
    data = getData("data/13800002122-15-t-s.txt")
    print(f"加载到原始 {len(data)} 个测试点")

    linear_segments = identify_linear_segments(data)
    print(f"识别出疑似公路轨迹段: {len(linear_segments)} 段")

    plot_linear_segments(data, linear_segments)