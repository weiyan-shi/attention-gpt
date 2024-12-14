import pandas as pd
import numpy as np
import cv2
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# 文件路径
asd_file = "dataset/pics/TrainingData/ASD/ASD_scanpath_28.txt"
td_file = "dataset/pics/TrainingData/TD/TD_scanpath_28.txt"
background_image_path = "dataset/pics/TrainingData/Images/28.png"

# 数据加载函数
def load_and_process_data(file_path, group_label):
    data = []
    current_subject_id = -1

    # 按行读取文件，分配 subject_id
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split(",")
            # 跳过列头
            if values[0].strip().lower() == "idx":
                continue
            # 新的 subject_id
            if values[0].strip() == "0":
                current_subject_id += 1
            data.append({
                "subject_id": current_subject_id,
                "x": int(values[1].strip()),
                "y": int(values[2].strip()),
                "duration": int(values[3].strip()),
                "group": group_label
            })
    return pd.DataFrame(data)

# 加载 ASD 和 TD 数据
asd_data = load_and_process_data(asd_file, "ASD")
td_data = load_and_process_data(td_file, "TD")

# 合并所有数据
combined_data = pd.concat([asd_data, td_data], ignore_index=True)

# 生成热图函数
def generate_heatmap(x, y, weights, img_shape, bandwidth=30):
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
threshold = np.percentile(normalized_heatmap, 90)  # 提取高于90%密度的区域
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

# 按注视点数量排序并选取最多的前 5 个区域
aoi_info = sorted(aoi_info, key=lambda x: x[1], reverse=True)[:5]

# 为每个注视点分配 AOI 标签
aoi_labels = np.full(len(combined_data), -1)  # 初始化所有点为 NONE 区域
for i, (contour, _) in enumerate(aoi_info):
    mask = np.zeros_like(binary_map, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    for idx, (x, y) in enumerate(zip(combined_data["x"], combined_data["y"])):
        if mask[int(y), int(x)] == 255:
            aoi_labels[idx] = i

# 将 AOI 标签添加到数据中
combined_data["aoi_label"] = aoi_labels

# 计算特征
def compute_features(data, cluster_centers):
    features = []
    for (subject_id, group), group_data in data.groupby(["subject_id", "group"]):
        subject_features = {"subject_id": subject_id, "group": group}
        none_duration = 0

        for cluster_id in range(len(cluster_centers)):
            # 获取属于该 AOI 的注视点
            cluster_data = group_data[group_data["aoi_label"] == cluster_id]
            fixation_count = len(cluster_data)
            total_duration = cluster_data["duration"].sum()
            mean_dwell_time = total_duration / fixation_count if fixation_count > 0 else 0

            # 保存 AOI 特征
            subject_features[f"AOI_{cluster_id}_FixationCount"] = fixation_count
            subject_features[f"AOI_{cluster_id}_TotalDuration"] = total_duration
            subject_features[f"AOI_{cluster_id}_MeanDwellTime"] = mean_dwell_time

        # 计算 NONE 区域的总注视时间
        cluster_data = group_data[group_data["aoi_label"] == -1]
        fixation_count = len(cluster_data)
        total_duration = cluster_data["duration"].sum()
        mean_dwell_time = total_duration / fixation_count if fixation_count > 0 else 0

        # 保存 NONE 特征
        subject_features[f"None_FixationCount"] = fixation_count
        subject_features[f"None_TotalDuration"] = total_duration
        subject_features[f"None_MeanDwellTime"] = mean_dwell_time
        features.append(subject_features)

    return pd.DataFrame(features)

# 计算所有特征
features = compute_features(combined_data, range(len(aoi_info)))

# 平均注视时长按群组聚合
grouped_features = features.groupby("group").mean()

# 打印结果
print("Individual Features:")
print(features.head())

print("\nGroup-Level Features (Mean):")
print(grouped_features)

# 可视化平均注视时长（每个 AOI 和 NONE 区域）
aoi_labels = [col for col in grouped_features.columns if col.startswith("AOI_") or col.startswith("None_")]
asd_means = grouped_features.loc["ASD", aoi_labels]
td_means = grouped_features.loc["TD", aoi_labels]

x = np.arange(len(aoi_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, asd_means, width, label="ASD", color="blue", alpha=0.7)
ax.bar(x + width/2, td_means, width, label="TD", color="orange", alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(aoi_labels, rotation=45)
ax.set_xlabel("AOIs")
ax.set_ylabel("Average Duration (ms)")
ax.set_title("Average Time Spent on AOIs and Outside (NONE)")
ax.legend()

plt.tight_layout()
plt.show()

grouped_features.to_csv("heatmap.csv", index=True)
