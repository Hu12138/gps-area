from flask import Flask, request, jsonify
import numpy as np
from shapely.geometry import LineString, MultiPoint
from pyproj import Geod, Transformer
from sklearn.cluster import DBSCAN
import logging

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlowingAreaCalculator:
    def __init__(self, tool_width=2.0, min_area=200):
        self.tool_width = tool_width
        self.min_area = min_area
        self.geod = Geod(ellps="WGS84")
        # 坐标转换器（WGS84转UTM）
        # 使用 UTM 50N (EPSG:32650)，请根据实际地理位置选择合适的UTM区域。
        # 如果您的点位可能跨越UTM区域，或者在其他区域，请务必修改此处的EPSG代码。
        # 可以使用 pyproj.get_utm_from_latlon(lat, lon) 来动态获取UTM区域，但这会增加计算开销。
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

    def calculate_areas(self, points):
        """
        计算耕作区域面积

        Args:
            points: 经过去重和预处理的GPS点列表 [[经度, 纬度], ...] - 注意：这里的输入已经是 [经度, 纬度] 格式了

        Returns:
            包含计算结果的字典或错误信息
        """
        try:
            # 1. 数据预处理 (去重和经纬度交换已在API层完成)
            points_array = np.array(points)
            logger.info(f"Received {len(points_array)} unique and correctly ordered points for processing")

            if len(points_array) < 3:
                 return {
                    "status": "error",
                    "message": "Not enough unique points (less than 3) after deduplication and coordinate swap"
                }

            # --- 校验输入点数据是否为有限数值 ---
            # 检查points_array中的所有元素是否都是有限的数字 (非NaN, 非inf)
            if not np.isfinite(points_array).all():
                 logger.error("Input points contain non-finite values (NaN, Inf) after deduplication and coordinate swap.")
                 return {
                    "status": "error",
                    "message": "Invalid input coordinate data detected after deduplication and coordinate swap. Contains non-finite values."
                }
            # --- 校验结束 ---

            # --- 打印去重后的点以便调试 ---
            logger.info(f"Unique points before transformation (lon, lat): {points_array.tolist()}")
            # --- 打印结束 ---


            # 2. 转换为UTM坐标（米制）
            # 注意：这里的转换假设所有点都在同一个UTM区域。如果跨区域，需要更复杂的处理。
            # 确保选择的UTM区域与实际点位匹配非常重要。
            try:
                # 这里的 transformer.transform 期望的是 (经度, 纬度)
                utm_points = np.array([self.transformer.transform(lon, lat) for lon, lat in points_array])
            except Exception as transform_error:
                 logger.error(f"Error during UTM transformation: {transform_error}")
                 return {
                    "status": "error",
                    "message": f"Error during coordinate transformation. Check if input points are valid and within the selected UTM zone (EPSG:32650). Details: {transform_error}"
                }

            # --- 打印转换后的UTM点以便调试 ---
            logger.info(f"Transformed UTM points before check: {utm_points.tolist()}")
            # --- 打印结束 ---

            # --- 校验转换后的UTM坐标是否为有限数值 ---
            # 检查utm_points中的所有元素是否都是有限的数字 (非NaN, 非inf)
            if not np.isfinite(utm_points).all():
                 logger.error("Transformed UTM points contain non-finite values (NaN, Inf).")
                 return {
                    "status": "error",
                    "message": "Coordinate transformation resulted in invalid values. Ensure input points are valid and within the selected UTM zone (EPSG:32650)."
                }
            # --- 校验结束 ---


            # 3. 动态聚类参数
            # 计算点间距中位数，用于自适应聚类参数
            median_dist = self._calc_median_distance(utm_points)
            # DBSCAN的eps参数：聚类半径，至少是工具宽度的3倍，或点间距中位数的2倍，取较大值
            eps = max(self.tool_width * 3, median_dist * 2)
            # DBSCAN的min_samples参数：形成一个簇所需的最小点数，至少为5，或总点数的3%
            min_samples = max(5, int(len(utm_points)*0.03))

            # 执行DBSCAN聚类
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(utm_points)
            logger.info(f"Clustering params: eps={eps:.1f}m, min_samples={min_samples}")
            # 记录噪声点比例
            logger.info(f"Noise points ratio: {np.mean(db.labels_ == -1):.1%}")

            # 4. 计算各簇面积
            results = {}
            # 遍历所有簇标签
            for label in set(db.labels_):
                if label == -1: continue  # 忽略噪声点 (标签为-1)

                # 获取当前簇的原始GPS点 (已经是 [经度, 纬度] 格式)
                cluster_points = points_array[db.labels_ == label]
                # 簇点数少于3个无法计算面积
                if len(cluster_points) < 3: continue

                # 使用凸包计算簇的面积
                # MultiPoint创建点集合，convex_hull计算凸包
                hull = MultiPoint(cluster_points).convex_hull
                # geod.geometry_area_perimeter计算几何对象的面积和周长
                area, _ = self.geod.geometry_area_perimeter(hull)
                area = abs(area) # 面积取绝对值

                # 如果面积大于最小面积阈值，则记录结果
                if area >= self.min_area:
                    results[f"field_{label}"] = {
                        "area_m2": round(area, 2), # 面积（平方米）
                        "area_mu": round(area/666.67, 2),  # 面积（亩）
                        "point_count": len(cluster_points) # 簇中的点数
                    }

            # 计算总面积
            total = sum(field["area_m2"] for field in results.values())
            response = {
                "status": "success",
                "field_count": len(results), # 耕作区域数量
                "total_area_m2": round(total, 2), # 总面积（平方米）
                "total_area_mu": round(total/666.67, 2), # 总面积（亩）
                "fields": results # 各个耕作区域的详细信息
            }

            logger.info(f"Processing completed. Found {len(results)} fields with total area {total:.2f}m²")
            return response

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _calc_median_distance(self, points):
        """
        计算点间距中位数 (在UTM坐标系下)

        Args:
            points: UTM坐标点数组

        Returns:
            点间距中位数
        """
        from scipy.spatial.distance import pdist
        if len(points) < 2: return 0
        # pdist计算点对之间的距离，median计算中位数
        return np.median(pdist(points))

@app.route('/calculate_plowing_area', methods=['POST'])
def calculate_plowing_area():
    """
    API端点：计算耕作面积
    请求格式：
    {
        "points": [[纬度, 经度], [纬度, 经度], ...], # GPS点列表，按时间顺序
        "tool_width": 2.0,  # 可选，默认2.0米
        "min_area": 200     # 可选，默认200平方米
    }
    """
    try:
        data = request.get_json()

        # 验证输入数据
        if not data or 'points' not in data:
            return jsonify({"status": "error", "message": "Missing 'points' in request"}), 400

        raw_points = data['points']
        if len(raw_points) < 3:
            return jsonify({"status": "error", "message": "At least 3 points are required"}), 400

        logger.info(f"Received {len(raw_points)} raw points.")

        # --- GPS点去重并保留顺序 ---
        # 设定一个小的容差值，用于比较浮点数（经纬度）。
        # 1e-6大约对应10厘米的距离（取决于纬度）。根据实际需求调整。
        tolerance = 1e-6
        temp_unique_points = [] # 临时存储去重后的 [纬度, 经度] 点
        if raw_points:
            temp_unique_points.append(raw_points[0]) # 总是保留第一个点
            for i in range(1, len(raw_points)):
                prev_point = raw_points[i-1]
                curr_point = raw_points[i]
                # 检查当前点与前一个点是否在容差范围内相同 (比较纬度和经度)
                if abs(curr_point[0] - prev_point[0]) > tolerance or \
                   abs(curr_point[1] - prev_point[1]) > tolerance:
                    temp_unique_points.append(curr_point)

        logger.info(f"After deduplication, {len(temp_unique_points)} unique points remain (lat, lon).")

        # 如果去重后点数不足，返回错误
        if len(temp_unique_points) < 3:
             return jsonify({"status": "error", "message": "Not enough unique points (less than 3) after deduplication"}), 400

        # --- 交换经纬度，转换为 [经度, 纬度] 格式 ---
        unique_points = [[point[1], point[0]] for point in temp_unique_points]
        logger.info(f"Swapped coordinates to [lon, lat] for {len(unique_points)} points.")
        # --- 交换结束 ---


        # 初始化计算器
        tool_width = float(data.get('tool_width', 2.0))
        min_area = float(data.get('min_area', 200))
        calculator = PlowingAreaCalculator(tool_width=tool_width, min_area=min_area)

        # 计算面积，传入去重并交换经纬度后的点
        result = calculator.calculate_areas(unique_points)

        return jsonify(result)

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # 在生产环境中请关闭 debug=True
    app.run(host='0.0.0.0', port=5000, debug=True)
