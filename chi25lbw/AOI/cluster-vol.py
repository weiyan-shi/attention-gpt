import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import cv2

# 文件路径
asd_file = "dataset/pics/TrainingData/ASD/ASD_scanpath_28.txt"
td_file = "dataset/pics/TrainingData/TD/TD_scanpath_28.txt"
background_image_path = "dataset/pics/TrainingData/Images/28.png"

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
labels = mean_shift.labels_

# 生成 Voronoi 图
vor = Voronoi(cluster_centers)

# 创建一个透明叠加图层
overlay = background_img.copy()

# 绘制 Voronoi 单元
for region in vor.regions:
    if not -1 in region and len(region) > 0:
        polygon = [vor.vertices[i] for i in region]
        # 确保顶点在图片范围内
        polygon = np.clip(polygon, [0, 0], [bg_width, bg_height]).astype(np.int32)
        if len(polygon) > 2:  # 只绘制有效多边形
            cv2.polylines(overlay, [polygon], isClosed=True, color=(255, 255, 0), thickness=2)

# 绘制注视点和聚类中心
for point, label in zip(fixation_points, labels[:len(fixation_points)]):  # 不绘制角点的标签
    cv2.circle(overlay, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)  # 注视点为绿色
for center in cluster_centers:
    cv2.drawMarker(overlay, (int(center[0]), int(center[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)  # 聚类中心为红色

# 显示结果
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")
# plt.title("AOI with Background Image (Including Borders)")
plt.show()
