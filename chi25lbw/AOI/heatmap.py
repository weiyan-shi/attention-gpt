import pandas as pd
import numpy as np
import cv2
from scipy.stats import gaussian_kde

# 文件路径
asd_file = "dataset/pics/TrainingData/ASD/ASD_scanpath_28.txt"  # 替换为实际 ASD 文件路径
td_file = "dataset/pics/TrainingData/TD/TD_scanpath_28.txt"  # 替换为实际 TD 文件路径
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
def generate_heatmap(x, y, weights, img_shape, bandwidth=80):
    # 构建二维网格
    x_grid, y_grid = np.linspace(0, img_shape[1], img_shape[1]), np.linspace(
        0, img_shape[0], img_shape[0]
    )
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # 使用高斯核密度估计 (KDE) 并引入权重
    values = np.vstack([x, y])
    kde = gaussian_kde(values, weights=weights, bw_method=bandwidth / max(img_shape))
    heatmap = np.reshape(kde(positions).T, xx.shape)
    return heatmap


# 加载背景图片
img = cv2.imread(background_image_path)

# 生成合并后的热图
heatmap = generate_heatmap(
    combined_data["x"],
    combined_data["y"],
    combined_data["duration"],
    img.shape,
    bandwidth=100,
)

# 归一化热图到 0-255
normalized_heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)

# 设置阈值分割高亮区域
threshold = np.percentile(normalized_heatmap, 60)  # 提取高于90%密度的区域
_, binary_map = cv2.threshold(normalized_heatmap, threshold, 255, cv2.THRESH_BINARY)

# 连通性分析，获取显著区域的轮廓
contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 筛选轮廓并统计注视点数量
aoi_info = []
for contour in contours:
    # 创建掩码
    mask = np.zeros_like(binary_map, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 统计轮廓内部的注视点数量
    points_inside = sum(
        [
            1
            for x, y in zip(combined_data["x"], combined_data["y"])
            if mask[int(y), int(x)] == 255
        ]
    )
    aoi_info.append((contour, points_inside))

# 按注视点数量排序
aoi_info = sorted(aoi_info, key=lambda x: x[1], reverse=True)[
    :5
]  # 选取注视点最多的前 5 个区域

# 创建透明叠加图层
overlay = img.copy()
for contour, points_inside in aoi_info:
    # 绘制轮廓区域并填充颜色
    if points_inside > 5:
        color = tuple(np.random.randint(0, 256, 3).tolist())  # 随机颜色
        cv2.drawContours(overlay, [contour], -1, color, thickness=cv2.FILLED)

# 添加透明度
alpha = 0.6  # 透明度（0完全透明，1完全不透明）
blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# 保存结果为 PNG 文件
cv2.imwrite("output_image_with_transparency.png", blended)
