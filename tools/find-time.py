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
data_time = []

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
                      # print(f'ados info in {filename}:')
                      # print(json.dumps(data['ados'], indent=4))
                      
                      # 提取眼动数据
                      eye_gaze = data['eye_gaze']
                      rx = eye_gaze.get('rx', [])
                      ry = eye_gaze.get('ry', [])
                      rz = eye_gaze.get('rz', [])

                      # 首先检查rx, ry, rz的长度是否一致
                      if len(rx) == len(ry) == len(rz):
                          data_time.append(len(rx))
                      else:
                        print(f'Error: Mismatched lengths in eye gaze data for file: {filename}')

                except json.JSONDecodeError as e:
                    print(f'Invalid JSON in file: {filename}. Error: {e}')
                except KeyError as e:
                    print(f'Missing key in file: {filename}. Error: {e}')

print('Search and plotting completed.')
