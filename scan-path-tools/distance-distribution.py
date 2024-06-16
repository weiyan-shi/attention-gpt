import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

# Ensure the TkAgg backend is used
plt.switch_backend('TkAgg')
plt.ion()  # Turn on interactive mode

def is_valid_point(x, y, z):
    return x is not None and y is not None and z is not None

folder_path = 'C:\\Users\\86178\\Desktop\\attention-gpt\\DREAMdataset'
output_folder_path = 'C:\\Users\\weiyan.shi\\Desktop\\attention-gpt\\output'

for root, dirs, files in os.walk(folder_path):
    for subdir in dirs:
        input_subfolder_path = os.path.join(folder_path, subdir)
        output_subfolder_path = os.path.join(output_folder_path, subdir)
        for filename in os.listdir(input_subfolder_path):
            if filename.endswith('.json'):
                with open(os.path.join(input_subfolder_path, filename), 'r', encoding='utf-8') as file:
                    try:
                        data = json.load(file)
                        if 'eye_gaze' in data and 'ados' in data and 'task' in data and 'ability' in data['task'] and data['task']['ability'] == 'JA':
                            print(f'ados info in {filename}:')
                            print(json.dumps(data['ados'], indent=4))
                            
                            eye_gaze = data['eye_gaze']
                            rx = eye_gaze.get('rx', [])
                            ry = eye_gaze.get('ry', [])
                            rz = eye_gaze.get('rz', [])

                            if len(rx) == len(ry) == len(rz):
                                task_info = data.get('task', {})
                                start_index = task_info.get('start', 0)
                                end_index = task_info.get('end', len(rx))
                                coords = [(x, y, z) for i, (x, y, z) in enumerate(zip(rx, ry, rz))
                                          if is_valid_point(x, y, z) and start_index <= i < end_index]
                                
                                if not coords:
                                    print(f"No valid eye gaze points in {filename}.")
                                    continue

                                # 计算当前文件中所有点之间的距离
                                distances = pdist(coords)

                                # 绘制当前文件的距离分布直方图
                                plt.figure()
                                plt.hist(distances, bins=50, alpha=0.7)
                                plt.xlabel('Distance')
                                plt.ylabel('Frequency')
                                plt.title(f'Distance Distribution in {filename}')
                                
                                # 保存或显示图表
                                output_filename = os.path.splitext(filename)[0] + '_distance_distribution.png'
                                output_filepath = os.path.join(output_subfolder_path, output_filename)
                                # if not os.path.exists(output_subfolder_path):
                                #     os.makedirs(output_subfolder_path)
                                # plt.savefig(output_filepath)
                                plt.show()

                    except json.JSONDecodeError as e:
                        print(f'Invalid JSON in file: {filename}. Error: {e}')
                    except KeyError as e:
                        print(f'Missing key in file: {filename}. Error: {e}')

print('Search and plotting completed.')
