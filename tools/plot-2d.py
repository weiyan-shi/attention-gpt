import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def is_valid_point(x, y, z):
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

                    if 'eye_gaze' in data and 'ados' in data and 'task' in data and 'ability' in data['task'] and data['task']['ability'] == 'JA':
                        eye_gaze = data['eye_gaze']
                        rx = eye_gaze.get('rx', [])
                        ry = eye_gaze.get('ry', [])
                        # ... [rest of the code for handling the mismatch and filtering points]
                        
                        task_info = data.get('task', {})
                        start_index = task_info.get('start', 0)
                        end_index = task_info.get('end', len(rx))
                        
                        # 过滤坐标点，只包含在start和end范围内的点
                        coords = [(x, y) for i, (x, y) in enumerate(zip(rx, ry))
                                  if x is not None and y is not None and start_index <= i < end_index]
                        
                        if not coords:
                            print(f"No valid points found in range for {filename}.")
                            continue
                        
                        rx_filtered, ry_filtered = zip(*coords)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))  # 创建一个足够大的2D图形
                        
                        # 根据需要可以设置更小的点大小和透明度
                        desired_size = 5  
                        desired_alpha = 0.1 
                        
                        ax.scatter(rx_filtered, ry_filtered, s=desired_size, alpha=desired_alpha)
                        
                        ados_items = list(data['ados']['preTest'].items())
                        midpoint = len(ados_items) // 2  # 找到中点位置
                        first_half = ", ".join([f"{k}: {v}" for k, v in ados_items[:midpoint]])
                        second_half = ", ".join([f"{k}: {v}" for k, v in ados_items[midpoint:]])
                        ados_info = f"ADOS info:{first_half}\n{second_half}"
                        print(ados_info)
                        plt.subplots_adjust(bottom=0.2)  # 调整下边距以留出批注空间
                        plt.figtext(0.5, 0.02, ados_info, fontsize=12, va="bottom", ha="center")  # 居中批注
                        
                        ax.set_xlabel('RX')
                        ax.set_ylabel('RY')
                        plt.title(f'Eye Gaze Data for {filename}')
                        
                        output_filename = os.path.splitext(filename)[0] + '_2D.png'
                        plt.savefig(os.path.join(output_subfolder_path, output_filename))
                        
                        plt.close(fig)

                except json.JSONDecodeError as e:
                    print(f'Invalid JSON in file: {filename}. Error: {e}')
                except KeyError as e:
                    print(f'Missing key in file: {filename}. Error: {e}')

print('Search and plotting completed.')
