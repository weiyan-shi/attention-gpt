import pandas as pd
import matplotlib.pyplot as plt

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
if "duration" not in asd_data.columns or "duration" not in td_data.columns:
    raise ValueError("The 'duration' column is missing in the dataset.")

# 添加分组标识符
asd_data["Group"] = (asd_data["Idx"] == 0).cumsum()
td_data["Group"] = (td_data["Idx"] == 0).cumsum()

# 计算质心函数
def calculate_centroids(data):
    centroids = []
    for _, group in data.groupby("Group"):
        total_duration = group["duration"].sum()
        centroid_x = (group["x"] * group["duration"]).sum() / total_duration
        centroid_y = (group["y"] * group["duration"]).sum() / total_duration
        centroids.append((centroid_x, centroid_y))
    return centroids

# def calculate_centroids(data):
#     centroids = []
#     for _, group in data.groupby("Group"):
#         centroid_x = group["x"].mean()
#         centroid_y = group["y"].mean()
#         centroids.append((centroid_x, centroid_y))
#     return centroids

# def calculate_centroids(data):
#     centroids = []
#     for _, group in data.groupby("Group"):
#         centroid_x = group["x"].median()
#         centroid_y = group["y"].median()
#         centroids.append((centroid_x, centroid_y))
#     return centroids

# def calculate_centroids(data):
#     centroids = []
#     for _, group in data.groupby("Group"):
#         # 计算 IQR
#         q1 = group["duration"].quantile(0.25)
#         q3 = group["duration"].quantile(0.75)
#         iqr = q3 - q1

#         # 剔除异常值
#         filtered_group = group[(group["duration"] >= q1 - 1.5 * iqr) & (group["duration"] <= q3 + 1.5 * iqr)]

#         total_duration = filtered_group["duration"].sum()
#         centroid_x = (filtered_group["x"] * filtered_group["duration"]).sum() / total_duration
#         centroid_y = (filtered_group["y"] * filtered_group["duration"]).sum() / total_duration
#         centroids.append((centroid_x, centroid_y))
#     return centroids




# 计算质心
asd_centroids = calculate_centroids(asd_data)
td_centroids = calculate_centroids(td_data)

# 加载背景图片
img = plt.imread(background_image_path)

# 添加背景图片
plt.imshow(img, aspect='auto')

# 绘制 ASD 组质心
for idx, (x, y) in enumerate(asd_centroids):
    plt.scatter(x, y, c="red", label="ASD" if idx == 0 else "", s=23)

# 绘制 TD 组质心
for idx, (x, y) in enumerate(td_centroids):
    plt.scatter(x, y, c="blue", label="TD" if idx == 0 else "", s=23)

# 图形装饰
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Fixation Centroids for ASD and TD Groups")
plt.legend()
plt.grid(False)

# 保存结果或显示图像
plt.savefig("chi25lbw/vis-result/28.png")  # 替换为实际保存路径
# plt.show()
