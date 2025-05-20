import numpy as np
from shapely.geometry import LineString
from pyproj import Transformer
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Agg')  # 必须放在导入pyplot之前
import matplotlib.pyplot as plt
from collections import defaultdict
from flask import Flask, request, jsonify
import os
from datetime import datetime

# 配置中文字体（根据系统调整）
matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class FarmWorkAnalyzer:
    def __init__(self, tool_width=2.0, min_points=10, min_area=100):
        self.tool_width = tool_width
        self.min_points = min_points
        self.min_area = min_area
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def analyze_work_areas(self, points, debug=False, plot_path=None):
        """分析耕作区域并返回结构化结果"""
        # 坐标转换
        proj_points = np.array([self.transformer.transform(float(lon), float(lat)) for lon, lat in points])

        # 聚类参数计算
        eps = self.tool_width * 1.5
        min_samples = max(2, int(len(points) * 0.005))

        # 执行DBSCAN聚类
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(proj_points)

        # 组织聚类结果
        clusters = defaultdict(list)
        for i, label in enumerate(db.labels_):
            if label != -1:
                clusters[label].append(proj_points[i])

        results = {}
        for label, cluster in clusters.items():
            # 过滤无效聚类
            if len(cluster) < self.min_points:
                continue
                
            # 排除直线路径
            if self._is_straight_path(cluster):
                if debug:
                    print(f"Cluster {label} rejected: straight path")
                continue

            # 计算区域面积
            line = LineString(cluster)
            area = line.buffer(self.tool_width / 2).area
            
            if area >= self.min_area:
                results[label] = {
                    'area': area,
                    'point_count': len(cluster)
                }

        # 生成调试图表
        if debug and plot_path:
            self._generate_debug_plot(proj_points, db.labels_, plot_path)

        return results, sum(info['area'] for info in results.values())

    def _is_straight_path(self, cluster, max_variance=10):
        """检测是否为直线路径"""
        if len(cluster) < 3:
            return True
            
        angles = []
        for i in range(1, len(cluster)-1):
            v1 = cluster[i] - cluster[i-1]
            v2 = cluster[i+1] - cluster[i]
            
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue
                
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
            angles.append(angle)
            
        return np.std(angles) < max_variance if angles else True

    def _generate_debug_plot(self, points, labels, save_path):
        """生成调试用图表"""
        plt.figure(figsize=(10, 6))
        unique_labels = set(labels)
        
        # 为每个聚类分配颜色
        colors = [plt.cm.tab10(i % 10) for i in unique_labels]
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            if label == -1:
                plt.scatter(points[mask, 0], points[mask, 1], 
                           c='gray', alpha=0.3, s=5, label='噪声点')
            else:
                plt.scatter(points[mask, 0], points[mask, 1],
                           c=[color], s=8, label=f'区域{label}')
        
        plt.legend()
        plt.title("耕作区域分析结果")
        plt.xlabel("投影坐标 X (米)")
        plt.ylabel("投影坐标 Y (米)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# 初始化Flask应用
app = Flask(__name__)
os.makedirs('img', exist_ok=True)  # 确保图片目录存在

@app.route('/calculate_plowing_area', methods=['POST'])
def calculate_plowing_area():
    try:
        # 解析请求数据
        data = request.json
        required_fields = ['points']
        if not all(field in data for field in required_fields):
            raise ValueError("缺少必填参数")

        # 参数处理
        points = [[float(lon), float(lat)] for lon, lat in data['points']]
        if len(points) < 2:
            raise ValueError("至少需要2个定位点")

        # 初始化分析器
        analyzer = FarmWorkAnalyzer(
            tool_width=data.get('tool_width', 2.0),
            min_points=data.get('min_points', 10),
            min_area=data.get('min_area', 100)
        )

        # 处理调试选项
        debug_mode = data.get('debug', False)
        plot_path = f"img/debug_{datetime.now().strftime('%Y%m%d%H%M%S')}.png" if debug_mode else None

        # 执行分析
        raw_results, total_area = analyzer.analyze_work_areas(
            points, 
            debug=debug_mode,
            plot_path=plot_path
        )

        # 格式化结果
        response = {
            "status": "success",
            "field_count": len(raw_results),
            "total_area_m2": round(total_area, 2),
            "total_area_mu": round(total_area / 666.67, 2),
            "fields": {}
        }

        # 填充各区域详情
        for idx, (label, info) in enumerate(raw_results.items()):
            field_key = f"field_{idx}"
            response["fields"][field_key] = {
                "area_m2": round(info['area'], 2),
                "area_mu": round(info['area'] / 666.67, 2),
                "point_count": info['point_count']
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
