import numpy as np
import matplotlib.pyplot as plt
from math import degrees
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ------------------ 工具函数 ------------------

EARTH_RADIUS = 6371000  # 地球半径（米）

def haversine_distance(p1, p2):
    """计算两点间大圆距离（米）"""
    lat1, lon1 = map(np.radians, p1)
    lat2, lon2 = map(np.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return EARTH_RADIUS * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def calculate_signed_angle(p1, p2, p3):
    """计算点p1->p2->p3的转向角度（正左负右）"""
    v1 = np.subtract(p2, p1)
    v2 = np.subtract(p3, p2)
    
    if haversine_distance(p1, p2) < 1 or haversine_distance(p2, p3) < 1:
        return 0.0

    angle = degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
    return (angle + 360) if angle < -180 else (angle - 360) if angle > 180 else angle

# ------------------ 核心类 ------------------

class TrajectoryAnalyzer:
    def __init__(self, points):
        """初始化，输入为经度在前或字符串格式的轨迹点"""
        self.raw_points = np.array(points)
        self.points = self._normalize_points(points)
        self.angles = self._calculate_angles()
        self.segments = []

    def _normalize_points(self, points):
        """标准化为纬度在前格式 [lat, lon]"""
        normed = []
        for p in points:
            lon, lat = (float(p[0]), float(p[1])) if isinstance(p[0], str) else (p[0], p[1])
            normed.append([lat, lon])
        return np.array(normed)

    def _calculate_angles(self):
        """预计算每个中间点的转向角"""
        return [
            calculate_signed_angle(self.points[i - 1], self.points[i], self.points[i + 1])
            for i in range(1, len(self.points) - 1)
        ]

    def analyze_segment(self, start, end):
        """分析子段的转向统计信息"""
        segment = self.points[start:end + 1]
        left, right, pattern = 0, 0, []

        for i in range(len(segment) - 2):
            angle = calculate_signed_angle(segment[i], segment[i + 1], segment[i + 2])
            if angle > 0:
                left += angle
                pattern.append('L')
            elif angle < 0:
                right += abs(angle)
                pattern.append('R')

        return {
            'left_total': left,
            'right_total': right,
            'net_rotation': left - right,
            'turning_pattern': ''.join(pattern),
            'segment_length': haversine_distance(segment[0], segment[-1])
        }

    def detect_u_turns(self, window=5, min_rotation=150):
        """检测掉头区域"""
        result = []
        for i in range(len(self.points) - window):
            analysis = self.analyze_segment(i, i + window - 1)
            dominant = max(analysis['left_total'], analysis['right_total']) / (analysis['left_total'] + analysis['right_total'] + 1e-6)
            pattern = analysis['turning_pattern']

            if abs(analysis['net_rotation']) > min_rotation and dominant > 0.7 and ('LLL' in pattern or 'RRR' in pattern):
                result.append({
                    'indices': (i, i + window - 1),
                    'center_point': self.points[i + window // 2],
                    'analysis': analysis
                })
        return result

    def segment_by_u_turns(self):
        """根据掉头点分段轨迹"""
        u_turns = self.detect_u_turns()
        if not u_turns:
            return [self.points]

        segments, last = [], 0
        for turn in u_turns:
            start = turn['indices'][0]
            if start > last:
                segments.append(self.points[last:start + 1])
            last = turn['indices'][1]
        if last < len(self.points):
            segments.append(self.points[last:])
        self.segments = segments
        return segments

    def calculate_curvature(self):
        """简单曲率指标：连续转向角绝对值差"""
        return [
            abs(self.angles[i+1] - self.angles[i])
            for i in range(len(self.angles) - 1)
        ]

    def find_straight_segments(self, window=5, max_deviation=10):
        """寻找近似直线段"""
        straight_segments = []
        for i in range(len(self.angles) - window):
            window_angles = self.angles[i:i + window]
            if max(abs(a) for a in window_angles) < max_deviation:
                straight_segments.append({'indices': (i, i + window)})
        return straight_segments

# ------------------ 可视化 ------------------

def enhanced_visualization(analyzer: TrajectoryAnalyzer):
    """综合轨迹图示"""
    plt.figure(figsize=(16, 12))

    # 轨迹基础图
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    lons, lats = analyzer.raw_points[:, 0], analyzer.raw_points[:, 1]
    ax1.plot(lons, lats, 'k-', alpha=0.5, label='轨迹')

    # 掉头点标记
    for turn in analyzer.detect_u_turns():
        lat, lon = turn['center_point']
        ax1.plot(lon, lat, '*', color='purple', markersize=12, label='掉头点')

    # 直线段标记
    for seg in analyzer.find_straight_segments():
        seg_pts = analyzer.points[seg['indices'][0]:seg['indices'][1] + 1]
        ax1.plot(seg_pts[:, 1], seg_pts[:, 0], 'g-', linewidth=3, alpha=0.7, label='直线段')

    ax1.set_title('轨迹特征识别')
    ax1.set_xlabel('经度')
    ax1.set_ylabel('纬度')
    ax1.legend()

    # 转向角度图
    ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    angles = analyzer.angles
    ax2.plot(range(1, len(analyzer.points) - 1), angles, 'b-', label='瞬时转向角')

    for i, a in enumerate(angles):
        if abs(a) > 30:
            ax2.plot(i + 1, a, 'ro')

    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_title('转向角度')
    ax2.set_xlabel('点序号')
    ax2.set_ylabel('角度')
    ax2.legend()

    # 曲率图
    ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    curvature = analyzer.calculate_curvature()
    ax3.plot(curvature, 'g-', label='曲率')
    for i, c in enumerate(curvature):
        if c > np.percentile(curvature, 90):
            ax3.plot(i, c, 'ro')

    ax3.set_title('轨迹曲率')
    ax3.set_xlabel('点序号')
    ax3.set_ylabel('曲率')
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 示例轨迹数据（经度在前，纬度在后）
    from getData import getData
    sample_points = getData("data/bug.json")
    
    # 初始化分析器
    analyzer = TrajectoryAnalyzer(sample_points)
    
    # 分段（基于掉头点）
    segments = analyzer.segment_by_u_turns()

    # 增强可视化分析
    enhanced_visualization(analyzer)

    # 或者使用更简单的可视化版本
    # plot_trajectory_analysis(analyzer)