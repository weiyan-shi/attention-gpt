import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from scipy.spatial import Voronoi
import cv2

# 文件路径
asd_file = "dataset/pics/TrainingData/ASD/ASD_scanpath_28.txt"
td_file = "dataset/pics/TrainingData/TD/TD_scanpath_28.txt"
background_image_path = "dataset/pics/TrainingData/Images/28.png"
output_image_path = "output_image_with_transparency.png"  # 输出文件路径

# 读取数据
asd_data = pd.read_csv(asd_file, delimiter=",")
td_data = pd.read_csv(td_file, delimiter=",")
asd_data.columns = asd_data.columns.str.strip()
td_data.columns = td_data.columns.str.strip()

# 检查列名
if "x" not in asd_data.columns or "y" not in asd_data.columns:
    raise ValueError("The 'x' and 'y' columns are missing in the ASD dataset.")
if "x" not in td_data.columns or "y" not in td_data.columns:
    raise ValueError("The 'x' and 'y' columns are missing in the TD dataset.")

# 合并数据
combined_data = pd.concat([asd_data, td_data], ignore_index=True)
fixation_points = combined_data[["x", "y"]].values

# 加载背景图片
background_img = cv2.imread(background_image_path)
bg_height, bg_width = background_img.shape[:2]

# 添加背景图片的四个角作为虚拟点
corner_points = np.array([[0, 0], [bg_width, 0], [0, bg_height], [bg_width, bg_height]])
augmented_points = np.vstack([fixation_points, corner_points])

# 使用均值漂移聚类生成全局 AOI
bandwidth = 100
mean_shift = MeanShift(bandwidth=bandwidth)
mean_shift.fit(augmented_points)

# 获取聚类中心和标签
cluster_centers = mean_shift.cluster_centers_

# 生成 Voronoi 图
vor = Voronoi(cluster_centers)

# 创建一个透明叠加图层
overlay = background_img.copy()

# 创建随机颜色用于填充 Voronoi 单元
np.random.seed(42)  # 固定随机种子以确保颜色一致
region_colors = [tuple(np.random.randint(0, 256, size=3).tolist()) for _ in range(len(vor.regions))]

# 绘制 Voronoi 单元并填充颜色
for region_index, region in enumerate(vor.regions):
    if not -1 in region and len(region) > 0:
        polygon = [vor.vertices[i] for i in region]
        # 确保顶点在图片范围内
        polygon = np.clip(polygon, [0, 0], [bg_width, bg_height]).astype(np.int32)
        if len(polygon) > 2:  # 只绘制有效多边形
            cv2.fillPoly(overlay, [polygon], color=region_colors[region_index])

# 处理外部区域（包含背景四个顶点）
for point in vor.ridge_points:
    if any(point >= len(cluster_centers)):  # 检查是否有角点
        external_polygon = [vor.vertices[i] for i in vor.regions if -1 not in i]
        external_polygon = np.clip(external_polygon, [0, 0], [bg_width, bg_height]).astype(np.int32)
        cv2.fillPoly(overlay, [external_polygon], color=(150, 150, 150))  # 灰色填充

# 添加透明度
alpha = 0.6  # 透明度：0完全透明，1完全不透明
blended = cv2.addWeighted(overlay, alpha, background_img, 1 - alpha, 0)

# 保存结果
cv2.imwrite(output_image_path, blended)
