import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 已知 AOI 的边界框信息
aoi_boxes = [
    {"class": 2, "confidence": 0.89, "bbox": [229.11, 233.88, 427.15, 296.05]},
    {"class": 0, "confidence": 0.88, "bbox": [436.85, 7.77, 1013.67, 767.00]},
]

# 加载 ASD 和 TD 数据
asd_file = "dataset/pics/TrainingData/ASD/ASD_scanpath_28.txt"
td_file = "dataset/pics/TrainingData/TD/TD_scanpath_28.txt"

def load_and_process_data(file_path, group_label):
    data = []
    current_subject_id = -1

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split(",")
            if values[0].strip().lower() == "idx":
                continue
            if values[0].strip() == "0":
                current_subject_id += 1
            data.append({
                "subject_id": current_subject_id,
                "x": float(values[1].strip()),
                "y": float(values[2].strip()),
                "duration": float(values[3].strip()),
                "group": group_label
            })
    return pd.DataFrame(data)

asd_data = load_and_process_data(asd_file, "ASD")
td_data = load_and_process_data(td_file, "TD")

combined_data = pd.concat([asd_data, td_data], ignore_index=True)

# 分配 AOI 标签
def assign_aoi_labels(data, aoi_boxes):
    aoi_labels = np.full(len(data), -1)
    point_counts = []
    for i, box in enumerate(aoi_boxes):
        x_min, y_min, x_max, y_max = box["bbox"]
        points_in_aoi = 0
        for idx, (x, y) in enumerate(zip(data["x"], data["y"])):
            if x_min <= x <= x_max and y_min <= y <= y_max:
                aoi_labels[idx] = i
                points_in_aoi += 1
        point_counts.append((i, points_in_aoi))
    # 根据注视点数量排序 AOI
    sorted_aoi = sorted(point_counts, key=lambda x: x[1], reverse=True)
    aoi_mapping = {old: new for new, (old, _) in enumerate(sorted_aoi)}
    aoi_labels = [aoi_mapping[label] if label != -1 else -1 for label in aoi_labels]
    return aoi_labels

combined_data["aoi_label"] = assign_aoi_labels(combined_data, aoi_boxes)

# 计算特征
def compute_features(data, aoi_count):
    features = []
    for (subject_id, group), group_data in data.groupby(["subject_id", "group"]):
        subject_features = {"subject_id": subject_id, "group": group}
        for cluster_id in range(aoi_count):
            cluster_data = group_data[group_data["aoi_label"] == cluster_id]
            fixation_count = len(cluster_data)
            total_duration = cluster_data["duration"].sum()
            mean_dwell_time = total_duration / fixation_count if fixation_count > 0 else 0

            subject_features[f"AOI_{cluster_id}_FixationCount"] = fixation_count
            subject_features[f"AOI_{cluster_id}_TotalDuration"] = total_duration
            subject_features[f"AOI_{cluster_id}_MeanDwellTime"] = mean_dwell_time

        none_data = group_data[group_data["aoi_label"] == -1]
        subject_features["None_FixationCount"] = len(none_data)
        subject_features["None_TotalDuration"] = none_data["duration"].sum()
        subject_features["None_MeanDwellTime"] = (
            none_data["duration"].sum() / len(none_data) if len(none_data) > 0 else 0
        )

        features.append(subject_features)
    return pd.DataFrame(features)

features = compute_features(combined_data, len(aoi_boxes))

# 平均注视时长按群组聚合
grouped_features = features.groupby("group").mean()

# 可视化
aoi_labels = [col for col in grouped_features.columns if col.startswith("AOI_") or col.startswith("None_")]
asd_means = grouped_features.loc["ASD", aoi_labels]
td_means = grouped_features.loc["TD", aoi_labels]

x = np.arange(len(aoi_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width / 2, asd_means, width, label="ASD", color="blue", alpha=0.7)
ax.bar(x + width / 2, td_means, width, label="TD", color="orange", alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(aoi_labels, rotation=45)
ax.set_xlabel("AOIs")
ax.set_ylabel("Average Duration (ms)")
ax.set_title("Average Time Spent on AOIs and Outside (NONE)")
ax.legend()

plt.tight_layout()
plt.show()

grouped_features.to_csv("semantic.csv", index=True)
