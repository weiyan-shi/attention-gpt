import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.stats import gaussian_kde

# 文件路径
asd_file = "dataset/pics/TrainingData/ASD/ASD_scanpath_28.txt"  # 替换为实际 ASD 文件路径
td_file = "dataset/pics/TrainingData/TD/TD_scanpath_28.txt"    # 替换为实际 TD 文件路径
background_image_path = "dataset/pics/TrainingData/Images/28.png"  # 背景图片路径

# 读取数据
asd_data = pd.read_csv(asd_file, delimiter=",")
td_data = pd.read_csv(td_file, delimiter=",")
asd_data.columns = asd_data.columns.str.strip()  # 去除列名空格
td_data.columns = td_data.columns.str.strip()

# 检查列名
if "x" not in asd_data.columns or "y" not in asd_data.columns:
    raise ValueError("The 'x' and 'y' columns are missing in the ASD dataset.")
if "x" not in td_data.columns or "y" not in td_data.columns:
    raise ValueError("The 'x' and 'y' columns are missing in the TD dataset.")

# 检查是否存在权重列（Duration）
if "duration" not in asd_data.columns:
    raise ValueError("The 'duration' column is missing in the ASD dataset.")
if "duration" not in td_data.columns:
    raise ValueError("The 'duration' column is missing in the TD dataset.")

# 合并 ASD 和 TD 数据
combined_data = pd.concat([asd_data, td_data], ignore_index=True)

# 生成热图函数
def generate_heatmap(x, y, weights, img_shape, bandwidth=30):
    # 构建二维网格
    x_grid, y_grid = np.linspace(0, img_shape[1], img_shape[1]), np.linspace(0, img_shape[0], img_shape[0])
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # 使用高斯核密度估计 (KDE) 并引入权重
    values = np.vstack([x, y])
    kde = gaussian_kde(values, weights=weights, bw_method=bandwidth / max(img_shape))
    heatmap = np.reshape(kde(positions).T, xx.shape)
    return heatmap

# 加载背景图片
img = plt.imread(background_image_path)

# 生成合并后的热图
heatmap = generate_heatmap(combined_data["x"], combined_data["y"], combined_data["duration"], img.shape, bandwidth=100)

# 归一化热图到 0-255
normalized_heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)

# 设置阈值分割高亮区域
threshold = np.percentile(normalized_heatmap, 90)  # 提取高于80%密度的区域
_, binary_map = cv2.threshold(normalized_heatmap, threshold, 255, cv2.THRESH_BINARY)

# 连通性分析，获取显著区域的轮廓
contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# min_area = 800  # 原本可能是100，改为10
# contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]


# 在原图上绘制区域
output_img = img.copy()
for contour in contours:
    # 计算边界框
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 用绿色框标注区域

# 保存热图和重点区域结果
plt.figure()
plt.imshow(img, aspect='auto', alpha=0.8)  # 显示背景图片
plt.imshow(heatmap, cmap="hot", alpha=0.5, extent=(0, img.shape[1], img.shape[0], 0))  # 热图
plt.title("Combined Fixation Heatmap")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.colorbar(label="Attention Intensity")
plt.savefig("chi25lbw/vis-result/28-heatmap.png")  # 替换为实际保存路径
plt.close()

plt.figure()
plt.imshow(output_img, aspect='auto')  # 绘制重点区域叠加图
plt.title("Key AOI Detection from Heatmap")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.savefig("chi25lbw/vis-result/28-aoi-detection.png")  # 替换为实际保存路径
plt.close()

# 输出区域信息
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    print(f"AOI at ({x}, {y}) with width {w} and height {h}")
