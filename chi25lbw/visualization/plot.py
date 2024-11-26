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
if "x" not in asd_data.columns or "y" not in asd_data.columns:
    raise ValueError("The 'x' and 'y' columns are missing in the ASD dataset.")
if "x" not in td_data.columns or "y" not in td_data.columns:
    raise ValueError("The 'x' and 'y' columns are missing in the TD dataset.")

# 添加分组标识符
asd_data["Group"] = (asd_data["Idx"] == 0).cumsum()
td_data["Group"] = (td_data["Idx"] == 0).cumsum()

# 打印分组数据（可选，便于调试）
def print_grouped_data(data, group_name):
    for group_id, group in data.groupby("Group"):
        print(f"\n{group_name} Group {group_id}:\n", group)

print_grouped_data(asd_data, "ASD")
print_grouped_data(td_data, "TD")

# 加载背景图片
img = plt.imread(background_image_path)

# 添加背景图片
plt.imshow(img, aspect='auto')

# 绘制 ASD 数据点
plt.scatter(asd_data["x"], asd_data["y"], c="red", label="ASD", s=23)

# 绘制 TD 数据点
plt.scatter(td_data["x"], td_data["y"], c="blue", label="TD", s=23)

# 图形装饰
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Fixation Points for ASD and TD Groups")
plt.legend()
plt.grid(False)

# 保存结果或显示图像
plt.savefig("chi25lbw/vis-result/28-pure.png")  # 替换为实际保存路径
# plt.show()
