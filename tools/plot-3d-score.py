import os
import json
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances, davies_bouldin_score, silhouette_score, calinski_harabasz_score
import numpy as np
import pandas as pd

def is_valid_point(x, y, z):
    # Check if any of the components are None or Null (which is interpreted as None in Python)
    return x is not None and y is not None and z is not None

# 设置要搜索的文件夹路径
folder_path = 'C:\\Users\\86178\\Desktop\\attention-gpt\\DREAMdataset'
output_folder_path = 'C:\\Users\\86178\\Desktop\\attention-gpt\\output'  # 保存的路径

ados_path = 'Data Pre-Post ADOS (all included).xlsx'
df = pd.read_excel(ados_path, usecols=['user_id', 'ADOS_Total_pre', 'ADOS_Total_post'])

# participant_data = {'user': [], 'filename': [], 'task': [], 'ados_total_pre':[], 'ados_total_post':[], 'd_db':[],'d_s':[],'d_ch':[], 'd_n_pc':[]}


# 遍历文件夹中的所有文件
for root, dirs, files in os.walk(folder_path):
    for subdir in dirs:
        # if subdir == 'User 3':
          participant_data = {'user': [], 'filename': [], 'condition': [], 'ados_total_pre':[], 'ados_total_post':[], 'd_db':[],'d_s':[],'d_ch':[], 'd_n_pc':[], 'condition':[], 'ability':[], 'difficulty_level':[]}
          input_subfolder_path = os.path.join(folder_path, subdir)
        #   output_subfolder_path = os.path.join(output_folder_path, subdir)
        #   if not os.path.exists(output_subfolder_path):
        #       os.makedirs(output_subfolder_path)
          for filename in os.listdir(input_subfolder_path):
            if filename.endswith('.json'):
              with open(os.path.join(input_subfolder_path, filename), 'r', encoding='utf-8') as file:
                  try:
                      data = json.load(file)
                  # 检查是否存在 "eye_gaze" 和 "ados" 字段
                      if 'eye_gaze' in data and 'ados' in data and 'task' in data and 'ability' in data['task']:
                      # 打印 "ados" 信息
                        # print(f'ados info in {filename}:')
                        # print(json.dumps(data['ados'], indent=4))
                        
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
                            clustering_data = list(zip(rx_filtered, ry_filtered, rz_filtered))

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
                            dbscan = DBSCAN(eps=0.07,  # 设置邻域半径
                                        min_samples=50)  # 设置成为核心对象所需的邻域样本数量
                            clusters = dbscan.fit_predict(clustering_data)

                            unique_labels = np.unique(clusters)
                            if len(unique_labels) < 2:
                                continue

                            # Davies-Bouldin Index
                            db_score = davies_bouldin_score(clustering_data, clusters)
                            participant_data['d_db'].append(db_score)

                            # Silhouette Score
                            silhouette_score_value = silhouette_score(clustering_data, clusters)
                            participant_data['d_s'].append(silhouette_score_value)

                            # Calinski-Harabasz Index
                            calinski_harabasz_score_value = calinski_harabasz_score(clustering_data, clusters)
                            participant_data['d_ch'].append(calinski_harabasz_score_value)

                            # Count noise points
                            noise_points_mask = clusters == -1
                            noise_indices = np.where(noise_points_mask)[0]
                            noise_points = clusters[noise_points_mask]
                            total_points = clusters.shape[0]

                            # Calculate noise percentage
                            noise_percentage = (len(noise_points) / total_points) * 100
                            participant_data['d_n_pc'].append(noise_percentage)

                            participant_data['ability'].append(data['task']['ability'])
                            participant_data['difficulty_level'].append(data['task']['difficultyLevel'])
                            participant_data['condition'].append(data['condition'])
                            user = int(subdir.split(' ')[1])
                            participant_data['user'].append(user)
                            participant_data['filename'].append(filename)
                            participant_data['ados_total_pre'].append(df[df['user_id'] == user]['ADOS_Total_pre'].item())
                            participant_data['ados_total_post'].append(df[df['user_id'] == user]['ADOS_Total_post'].item())
                      else:
                          print(f'Error: Mismatched lengths in eye gaze data for file: {filename}')

                  except json.JSONDecodeError as e:
                      print(f'Invalid JSON in file: {filename}. Error: {e}')
                  except KeyError as e:
                      print(f'Missing key in file: {filename}. Error: {e}')
          result_df = pd.DataFrame(participant_data)
          # Convert to CSV
          result_df.to_csv('score/' + subdir + '.csv', index=False)

print('Search and plotting completed.')
