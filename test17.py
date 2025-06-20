import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import json
from datetime import datetime
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)

def haversine(p1, p2):
    """计算两个经纬度点之间的距离（米）"""
    lon1, lat1, lon2, lat2 = map(np.radians, [p1["lon"], p1["lat"], p2["lon"], p2["lat"]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))

def getData(path):
    """加载并预处理数据"""
    with open(path, "r") as f:
        raw = json.load(f)
    raw = raw.get("points")
    
    points = []
    for p in raw:
        if "lon" in p and "lat" in p:
            try:
                point = {
                    "lon": float(p["lon"]),
                    "lat": float(p["lat"]),
                    "speed": float(p.get("speed", 0)),
                    "time": datetime.strptime(p["time"], "%Y-%m-%d %H:%M:%S")
                }
                points.append(point)
            except (ValueError, KeyError) as e:
                print(f"数据格式错误: {e}, 跳过该点")
    
    # 按时间排序
    points.sort(key=lambda x: x["time"])
    return points

def calculate_segment_features(segment):
    """计算轨迹段的特征"""
    features = {
        "length": 0,
        "duration": 0,
        "avg_speed": 0,
        "straightness": 0,
        "point_count": len(segment)
    }
    
    if len(segment) < 2:
        return features
    
    # 计算总长度和持续时间
    total_length = 0
    for i in range(1, len(segment)):
        dist = haversine(segment[i-1], segment[i])
        time_diff = (segment[i]["time"] - segment[i-1]["time"]).total_seconds()
        total_length += dist
        features["duration"] += time_diff
    
    features["length"] = total_length
    features["avg_speed"] = total_length / features["duration"] if features["duration"] > 0 else 0
    
    # 计算直线度 (起点到终点的距离与总长度的比值)
    start_end_dist = haversine(segment[0], segment[-1])
    features["straightness"] = start_end_dist / total_length if total_length > 0 else 0
    
    return features

def identify_non_operation_segments(points, config):
    """改进的非作业轨迹识别"""
    non_op_segments = []
    current_segment = []
    
    # 预处理：坐标转换和计算基本属性
    for i, p in enumerate(points):
        try:
            p["x"], p["y"] = transformer.transform(p["lon"], p["lat"])
            if i > 0:
                prev_p = points[i-1]
                p["dist_from_prev"] = haversine(prev_p, p)
                p["time_from_prev"] = (p["time"] - prev_p["time"]).total_seconds()
                p["speed_from_prev"] = p["dist_from_prev"] / p["time_from_prev"] if p["time_from_prev"] > 0 else 0
        except Exception as e:
            print(f"预处理错误: {e}")
            continue
    
    for i in range(1, len(points)):
        p_prev, p_curr = points[i-1], points[i]
        
        # 检查是否满足连续移动条件
        if (p_curr.get("dist_from_prev", 0) > config["min_step_length"] and 
            p_curr.get("time_from_prev", float('inf')) < config["max_step_time"] and 
            p_curr.get("speed_from_prev", 0) > config["min_step_speed"]):
            
            if not current_segment:
                current_segment.append(p_prev)
            current_segment.append(p_curr)
        else:
            if current_segment:
                features = calculate_segment_features(current_segment)
                
                # 调试输出
                print(f"检查段: 点数={features['point_count']}, 长度={features['length']:.1f}m, "
                      f"时长={features['duration']:.1f}s, 均速={features['avg_speed']:.1f}m/s, "
                      f"直线度={features['straightness']:.2f}")
                
                if (features["point_count"] >= config["min_segment_points"] and
                    features["length"] >= config["min_total_length"] and
                    features["avg_speed"] >= config["min_avg_speed"] and
                    features["straightness"] >= config["min_straightness"]):
                    
                    non_op_segments.append(current_segment)
                    print("--> 识别为非作业段")
                
                current_segment = []
    
    # 检查最后一段
    if current_segment:
        features = calculate_segment_features(current_segment)
        if (features["point_count"] >= config["min_segment_points"] and
            features["length"] >= config["min_total_length"] and
            features["avg_speed"] >= config["min_avg_speed"] and
            features["straightness"] >= config["min_straightness"]):
            
            non_op_segments.append(current_segment)
    
    return non_op_segments

def plot_segments(points, segments, title="非作业轨迹识别结果"):
    """可视化轨迹和识别结果"""
    plt.figure(figsize=(12, 8))
    
    # 所有点
    all_x = [p.get("x", 0) for p in points]
    all_y = [p.get("y", 0) for p in points]
    plt.plot(all_x, all_y, c='gray', linewidth=1, alpha=0.3, label="全部轨迹")
    
    # 高亮非作业轨迹段
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    for i, seg in enumerate(segments):
        seg_x = [p.get("x", 0) for p in seg]
        seg_y = [p.get("y", 0) for p in seg]
        plt.plot(seg_x, seg_y, linewidth=2.5, color=colors[i], label=f"非作业段{i+1}")
        
        # 标记起点和终点
        plt.scatter(seg[0].get("x", 0), seg[0].get("y", 0), c='green', marker='o', s=50, zorder=3)
        plt.scatter(seg[-1].get("x", 0), seg[-1].get("y", 0), c='red', marker='x', s=50, zorder=3)
    
    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 配置参数
CONFIG = {
    "min_segment_points": 5,       # 最小段点数
    "min_total_length": 100,       # 最小总长度(米)
    "min_avg_speed": 2,           # 最小平均速度(m/s)
    "min_step_length": 1,          # 最小步长(米)
    "max_step_time": 120,          # 最大步长时间(秒)
    "min_step_speed": 0.5,         # 最小步速(m/s)
    "min_straightness": 0.6        # 最小直线度(0-1)
}

if __name__ == "__main__":
    # 测试数据路径
    data_path = "data/13885004840-11-t-s.json"
    data_path = "data/13800002122-14-t-s.txt"
    data_path = "data/13800002122-15-t-s.txt"
    
    # 加载数据
    data = getData(data_path)
    print(f"加载到 {len(data)} 个轨迹点")
    
    if not data:
        print("错误: 没有加载到有效数据!")
        exit()
    
    # 打印前几个点检查数据
    print("\n前5个数据点示例:")
    for i, p in enumerate(data[:5]):
        print(f"{i+1}. 经度={p['lon']:.6f}, 纬度={p['lat']:.6f}, 速度={p.get('speed', 0):.1f}m/s, 时间={p['time']}")
    
    # 识别非作业段
    print("\n开始识别非作业轨迹...")
    non_op_segments = identify_non_operation_segments(data, CONFIG)
    print(f"\n识别出 {len(non_op_segments)} 段非作业轨迹")
    
    # 如果没有识别到，尝试放宽参数
    if len(non_op_segments) == 0:
        print("\n尝试放宽识别参数...")
        relaxed_config = CONFIG.copy()
        relaxed_config.update({
            "min_segment_points": 3,
            "min_total_length": 50,
            "min_avg_speed": 1,
            "min_step_speed": 0.3,
            "min_straightness": 0.4
        })
        non_op_segments = identify_non_operation_segments(data, relaxed_config)
        print(f"放宽参数后识别出 {len(non_op_segments)} 段非作业轨迹")
    
    # 可视化结果
    if non_op_segments:
        plot_segments(data, non_op_segments)
    else:
        print("\n警告: 未能识别出任何非作业轨迹段!")
        print("可能原因:")
        print("1. 数据中没有明显的非作业轨迹")
        print("2. 参数设置仍然过于严格")
        print("3. 数据格式或质量问题")
        
        # 绘制原始轨迹以供检查
        plt.figure(figsize=(12, 8))
        x = [p.get("x", 0) for p in data]
        y = [p.get("y", 0) for p in data]
        plt.plot(x, y, c='blue', linewidth=1, alpha=0.5)
        plt.title("原始轨迹数据")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid(True, alpha=0.3)
        plt.show()