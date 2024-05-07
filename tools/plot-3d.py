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
folder_path = 'C:\\Users\\weiyan.shi\\Desktop\\attention-gpt\\dataset\\user-3'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为.json
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            try:
                # 加载JSON数据
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

                        desired_size = 5  # 调整点大小
                        desired_alpha = 0.1  # 调整透明度，0.5表示半透明，可以根据需要调整


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

                        # 使用DBSCAN对眼动数据进行聚类
                        db = DBSCAN(eps=0.07,  # 设置邻域半径
                                    min_samples=80)  # 设置成为核心对象所需的邻域样本数量
                        clusters = db.fit_predict(X)

                        # 创建三维绘图
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')

                        # 绘制聚类后的点
                        scatter = ax.scatter(rx_filtered, ry_filtered, rz_filtered, c=clusters, s=desired_size, alpha=desired_alpha, cmap='viridis', label=clusters)

                        # 设置标签
                        ax.set_xlabel('RX')
                        ax.set_ylabel('RY')
                        ax.set_zlabel('RZ')

                        # 显示图例
                        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                        ax.add_artist(legend1)

                        # 显示图形
                        plt.title(f'Eye Gaze Data with DBSCAN Clustering for {filename}')
                        plt.show()

                    else:
                        print(f'Error: Mismatched lengths in eye gaze data for file: {filename}')

            except json.JSONDecodeError as e:
                print(f'Invalid JSON in file: {filename}. Error: {e}')
            except KeyError as e:
                print(f'Missing key in file: {filename}. Error: {e}')

print('Search and plotting completed.')
