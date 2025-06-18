import matplotlib.pyplot as plt
import random
from math import atan2, degrees
import matplotlib
from pyproj import Transformer


matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def calculate_turning_angle(p1, p2, p3):
    """计算三点形成的转角（0-180度）"""
    # 向量AB (lon变化, lat变化)
    v1 = (p2[1]-p1[1], p2[0]-p1[0])
    # 向量BC
    v2 = (p3[1]-p2[1], p3[0]-p2[0])
    
    # 计算夹角（使用atan2避免数值问题）
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    angle_rad = atan2(abs(cross), dot)
    return degrees(angle_rad)

def segment_track(points, window_size=6, angle_threshold=150):
    """基于滑动窗口总转角的分割算法"""
    if len(points) < window_size:
        return [points]
    
    segments = []
    split_positions = []
    
    # 第一轮：检测所有潜在分割点
    for i in range(len(points) - window_size + 1):
        window_points = points[i:i+window_size]
        total_angle = 0
        
        for j in range(len(window_points)-2):
            angle = calculate_turning_angle(window_points[j], 
                                          window_points[j+1], 
                                          window_points[j+2])
            total_angle += angle
        
        if total_angle > angle_threshold:
            print(f"窗口里面的GPS：{window_points}")
            split_pos = i + (window_size // 2)
            if split_pos not in split_positions:
                split_positions.append(split_pos)
    
    # 处理分割点
    if not split_positions:
        return [points]
    
    split_positions.sort()
    prev_pos = 0
    for pos in split_positions:
        if pos > prev_pos and pos < len(points):
            segments.append(points[prev_pos:pos+1])
            prev_pos = pos
    
    if prev_pos < len(points):
        segments.append(points[prev_pos:])
    
    return segments

def generate_random_color():
    """生成鲜艳的随机颜色"""
    return (random.random(), random.random(), random.random())

def visualize(points, segments):
    """改进的可视化函数"""
    plt.figure(figsize=(12, 8), dpi=100)
    
    # 绘制所有原始点（半透明）
    lons, lats = zip(*points)
    plt.plot(lons, lats, 'ko', markersize=3, alpha=0.3, label='GPS点')
    
    # 绘制每个线段（不同颜色）
    for i, seg in enumerate(segments):
        seg_lons, seg_lats = zip(*seg)
        color = generate_random_color()
        
        # 主线段
        plt.plot(seg_lons, seg_lats, 
                color=color,
                linewidth=3,
                marker='o',
                markersize=5,
                markeredgecolor='k',
                label=f'线段{i+1}')
        
        # 线段起点标记
        plt.plot(seg_lons[0], seg_lats[0], 
                'o', color=color, markersize=8)
        
        # 线段终点标记（如果是分割点）
        if i < len(segments)-1:
            plt.plot(seg_lons[-1], seg_lats[-1], 
                    'rx', markersize=12, mew=2)
    
    # 自动调整坐标范围（增加5%边距）
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    lon_margin = (max_lon - min_lon) * 0.05
    lat_margin = (max_lat - min_lat) * 0.05
    
    plt.xlim(min_lon - lon_margin, max_lon + lon_margin)
    plt.ylim(min_lat - lat_margin, max_lat + lat_margin)
    
    plt.title(f"轨迹分割结果（共{len(segments)}段）", fontsize=14)
    plt.xlabel("经度", fontsize=12)
    plt.ylabel("纬度", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 优化图例显示
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc='best',
              fontsize=10,
              framealpha=0.8)
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # 测试轨迹
    from getData import getData
    test_points = getData(r"data/线段测试.json")
    test_points = [ [float(p[1]),float(p[0])] for p in test_points]
    # 执行分割算法
    segments = segment_track(test_points, window_size=6, angle_threshold=150)
    
    # 输出分割结果
    print(f"原始点数：{len(test_points)}")
    print(f"分割段数：{len(segments)}")
    # for i, seg in enumerate(segments):
    #     print(f"线段{i+1}: {len(seg)}个点 | 起点：{seg[0]} | 终点：{seg[-1]}")
    
    # 可视化
    visualize(test_points, segments)
