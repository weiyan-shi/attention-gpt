import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

def is_valid_point(x, y, z):
    # Check if any of the components are None or Null (which is interpreted as None in Python)
    return x is not None and y is not None and z is not None

# 设置要搜索的文件夹路径
folder_path = 'C:\\Users\\weiyan.shi\\Desktop\\attention-gpt\\DREAMdataset'
output_folder_path = 'C:\\Users\\weiyan.shi\\Desktop\\attention-gpt\\output'  # 图片保存的路径

# 遍历文件夹中的所有文件
for root, dirs, files in os.walk(folder_path):
    for subdir in dirs:
        input_subfolder_path = os.path.join(folder_path, subdir)
        output_subfolder_path = os.path.join(output_folder_path, subdir)
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)
        for filename in os.listdir(input_subfolder_path):
          if filename.endswith('.json'):
            with open(os.path.join(input_subfolder_path, filename), 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                # 检查是否存在 "eye_gaze" 和 "ados" 字段
                    if 'eye_gaze' in data and 'ados' in data and 'task' in data and 'ability' in data['task'] and data['task']['ability'] == 'JA':
                    # 打印 "ados" 信息
                      print(f'ados info in {filename}:')
                      print(json.dumps(data['ados'], indent=4))
                      
                      # 提取眼动数据
                      eye_gaze = data['eye_gaze']
                      rx = eye_gaze.get('rx', [])
                      ry = eye_gaze.get('ry', [])
                      rz = eye_gaze.get('rz', [])

                      # 首先检查rx, ry, rz的长度是否一致
                      if len(rx) == len(ry) == len(rz):
                          # 使用列表推导式和zip来过滤掉任何包含None的点
                          task_info = data.get('task', {})
                          start_index = task_info.get('start', 0)  # 默认值为0
                          end_index = task_info.get('end', len(rx))  # 默认值为rx的长度
                          coords = [(x, y, z) for i, (x, y, z) in enumerate(zip(rx, ry, rz))
                                    if is_valid_point(x, y, z) and start_index <= i < end_index]
                          
                          if not coords:
                              print(f"No valid eye gaze points in {filename}.")
                              continue

                          # 解压缩坐标列表以进行绘图
                          rx_filtered, ry_filtered, rz_filtered = zip(*coords)

                          # 将坐标转换成2D数组形式，这是scikit-learn中DBSCAN算法的要求格式
                          X = list(zip(rx_filtered, ry_filtered, rz_filtered))

                          desired_size = 20  # 调整点大小
                          desired_alpha = 0.7  # 调整透明度，0.5表示半透明，可以根据需要调整


                          # neigh = NearestNeighbors(n_neighbors=80)  # min_samples通常设置为MinPts，这里取2做示例
                          # nbrs = neigh.fit(X)
                          # distances, indices = nbrs.kneighbors(X)

                          # # 按距离排序
                          # distances = np.sort(distances, axis=0)
                          # distances = distances[:, 1]  # 取出第二近的距离，因为第一近的距离总是0

                          # plt.plot(distances)
                          # plt.title("K-Distance Graph")
                          # plt.xlabel("Points sorted by distance")
                          # plt.ylabel("Kth nearest distance")
                          # plt.show()

                          rx_filtered = np.array(rx_filtered)
                          ry_filtered = np.array(ry_filtered)
                          rz_filtered = np.array(rz_filtered)

                          # 使用DBSCAN对眼动数据进行聚类
                          db = DBSCAN(eps=0.07,  # 设置邻域半径
                                      min_samples=50)  # 设置成为核心对象所需的邻域样本数量
                          clusters = db.fit_predict(X)
                          mask = clusters != -1

                          rx_non_noise = rx_filtered[mask]
                          ry_non_noise = ry_filtered[mask]
                          rz_non_noise = rz_filtered[mask]
                          clusters_non_noise = clusters[mask]


                          # 创建三维绘图
                          fig, ax = plt.subplots(figsize=(3, 2))  # 创建一个足够大的2D图形

                          # 绘制聚类后的点
                          scatter = ax.scatter(
                              rx_non_noise,
                              ry_non_noise,
                              # rz_non_noise,
                              c=clusters_non_noise,   # Apply the mask to the clusters array as well.
                              s=desired_size,
                              alpha=desired_alpha,
                              # cmap='viridis'
                          )
                          # 设置标签
                          ax.set_xlabel('RX')
                          ax.set_ylabel('RY')
                          # plt.axis('off')
                          plt.xticks([])
                          plt.yticks([])
                          # ax.set_zlabel('RZ')

                          # 显示图例
                          # legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                          # ax.add_artist(legend1)

                          ados_items = list(data['ados']['preTest'].items())
                          midpoint = len(ados_items) // 2  # 找到中点位置
                          # first_half = ", ".join([f"{k}: {v}" for k, v in ados_items[:midpoint]])
                          # second_half = ", ".join([f"{k}: {v}" for k, v in ados_items[midpoint:]])
                          # ados_info = f"ADOS info:{first_half}\n{second_half}"
                          ados_info = f"ADOS info:{ados_items[7]}"
                          print(ados_info)
                          # plt.subplots_adjust(bottom=0.2)  # 调整下边距以留出批注空间
                          # plt.figtext(0.5, 0.02, ados_info, fontsize=12, va="bottom", ha="center")  # 居中批注
                          # plt.title(f'Eye Gaze Data with DBSCAN Clustering for {filename}')
                          # plt.show()
                          output_filename = os.path.splitext(filename)[0] + '_2D.png'
                          plt.savefig(os.path.join(output_subfolder_path, output_filename))
                          # 显示图形
                          # plt.show()

                    else:
                        print(f'Error: Mismatched lengths in eye gaze data for file: {filename}')

                except json.JSONDecodeError as e:
                    print(f'Invalid JSON in file: {filename}. Error: {e}')
                except KeyError as e:
                    print(f'Missing key in file: {filename}. Error: {e}')

print('Search and plotting completed.')
