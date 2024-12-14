import pandas as pd
import numpy as np
import cv2
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

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

# 显著性图相关方法
def resize_to_even(image):
    """调整图像尺寸为偶数"""
    h, w = image.shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return cv2.resize(image, (new_w, new_h))

def manual_pyr_up(image, target_size):
    """手动调整 pyrUp 输出尺寸"""
    upscaled = cv2.pyrUp(image)
    return cv2.resize(upscaled, target_size)

def itti_saliency_map(image):
    """计算 Itti-Koch 显著性图"""
    # 转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯金字塔
    pyr = [resize_to_even(gray)]  # 确保起始图像为偶数
    for i in range(1, 6):  # 创建 6 层金字塔
        down = cv2.pyrDown(pyr[i - 1])
        pyr.append(resize_to_even(down))  # 确保每层尺寸为偶数
    
    # 中心环对比 (Center-Surround Differences)
    cs_maps = []
    for i in range(2, 5):  # 高金字塔层
        center = pyr[i]
        for delta in range(1, 3):  # 环形差异
            if i + delta < len(pyr):  # 防止越界
                surround = pyr[i + delta]
                target_size = (center.shape[1], center.shape[0])  # 恢复到中心尺寸
                surround_resized = manual_pyr_up(surround, target_size)  # 使用手动调整
                cs_maps.append(cv2.absdiff(center, surround_resized))
    
    # 合并显著性图
    saliency_map = np.zeros_like(gray, dtype=np.float32)
    for m in cs_maps:
        resized_m = cv2.resize(m, gray.shape[::-1])  # 恢复到原始输入图像尺寸
        saliency_map += resized_m.astype(np.float32)
    
    # 归一化到 0-255
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    return saliency_map.astype(np.uint8)

def generate_aoi_overlay(image, saliency_map, min_area=1000, top_n=5):
    """从显著性图生成前 N 个面积最大的 AOI，并根据注视点数量排序"""
    # 使用 Otsu 阈值分割显著性区域
    threshold = threshold_otsu(saliency_map)
    _, binary_map = cv2.threshold(saliency_map, threshold, 255, cv2.THRESH_BINARY)
    
    # 提取轮廓
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 筛选轮廓：剔除小面积区域
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    # 统计每个轮廓内部的注视点数量
    contour_info = []
    for contour in filtered_contours:
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        points_inside = sum(
            1
            for x, y in zip(combined_data["x"], combined_data["y"])
            if mask[int(y), int(x)] == 255
        )
        contour_info.append((contour, points_inside))

    # 按注视点数量排序并选择前 N 个
    sorted_contours = sorted(contour_info, key=lambda x: x[1], reverse=True)[:top_n]
    
    return [contour for contour, _ in sorted_contours]

# 主程序
image = cv2.imread(background_image_path)
if image is None:
    raise ValueError("Failed to load the background image.")

# 调整输入图像尺寸为偶数
image = resize_to_even(image)

# 计算 Itti Saliency Map
itti_saliency = itti_saliency_map(image)

# 获取前 5 个面积最大的 AOI
contours = generate_aoi_overlay(image, itti_saliency, min_area=1000, top_n=5)

# 为注视点分配 AOI 标签
def assign_aoi_labels(data, contours):
    aoi_labels = np.full(len(data), -1)  # 初始化所有点为 NONE 区域
    for i, contour in enumerate(contours):
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        for idx, (x, y) in enumerate(zip(data["x"], data["y"])):
            if mask[int(y), int(x)] == 255:
                aoi_labels[idx] = i
    return aoi_labels

# 分配 AOI 标签
combined_data["aoi_label"] = assign_aoi_labels(combined_data, contours)

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
features = compute_features(combined_data, range(len(contours)))

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

grouped_features.to_csv("saliency.csv", index=True)
