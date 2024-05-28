import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# 设定包含CSV文件的文件夹路径
folder_path = 'score'

# 初始化一个空的DataFrame列表来存放每个文件的数据
dfs = []

# 遍历文件夹内的每一个文件
for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为.csv
    if filename.endswith('.csv'):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 读取CSV文件，并添加到列表中
        dfs.append(pd.read_csv(file_path))

# 使用concat函数合并所有的DataFrame对象
df = pd.concat(dfs, ignore_index=True)


# 进一步分割出包含 'Initial diagnosis' 和 'Intervention' 的两组数据
# df_initial = df_ja[df_ja['filename'].str.contains('Initial diagnosis')]
# df_intervention = df_ja[df_ja['filename'].str.contains('Intervention')]

# 对于 'd_db', 'd_s', 'd_ch', 'd_n_pc' 执行 t-test 并获取 p 值
columns = ['d_db', 'd_s', 'd_ch', 'd_n_pc']

# for col in columns:
#     plt.figure()
#     plt.scatter(df[col], df['ados_total_pre'])
#     plt.title(f"ADOS Pre vs {col}")
#     plt.xlabel(col)
#     plt.ylabel('ADOS Pre Score')
#     plt.show()

grouped_df = df.groupby('ados_total_pre')[['d_db', 'd_s', 'd_ch', 'd_n_pc']].mean()

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 10))

# 设置标题
fig.suptitle('Average Clustering Metrics vs. ADOS Total Pre Score')

# Davies-Bouldin Index (d_db)
axes[0].plot(grouped_df.index, grouped_df['d_db'], marker='o', color='blue')
axes[0].set_title('Davies-Bouldin Index (d_db)')
axes[0].grid(True)

# Silhouette Score (d_s)
axes[1].plot(grouped_df.index, grouped_df['d_s'], marker='s', color='green')
axes[1].set_title('Silhouette Score (d_s)')
axes[1].grid(True)

# Calinski-Harabasz Index (d_ch)
axes[2].plot(grouped_df.index, grouped_df['d_ch'], marker='^', color='red')
axes[2].set_title('Calinski-Harabasz Index (d_ch)')
axes[2].grid(True)

# Percentage of Noise Points (d_n_pc)
axes[3].plot(grouped_df.index, grouped_df['d_n_pc'], marker='x', color='purple')
axes[3].set_title('Percentage of Noise Points (d_n_pc)')
axes[3].grid(True)

# 设置横轴标签
for ax in axes:
    ax.set_xlabel('ADOS Total Pre Score')

# 设置纵轴标签
axes[0].set_ylabel('Index Value')
axes[1].set_ylabel('Score')
axes[2].set_ylabel('Index Value')
axes[3].set_ylabel('Percentage')

# 自动调整子图参数以给定填充物
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 显示图形
plt.show()
print(grouped_df)

