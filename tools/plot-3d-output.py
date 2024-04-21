import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def is_valid_point(x, y, z):
    return x is not None and y is not None and z is not None

# 设置要搜索的文件夹路径
folder_path = 'C:\\Users\\weiyan.shi\\Desktop\\attention-gpt\\dataset\\user-3'
output_folder_path = 'C:\\Users\\weiyan.shi\\Desktop\\attention-gpt\\output\\user-3'  # 图片保存的路径

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)

                if 'eye_gaze' in data and 'ados' in data:
                    eye_gaze = data['eye_gaze']
                    rx = eye_gaze.get('rx', [])
                    ry = eye_gaze.get('ry', [])
                    rz = eye_gaze.get('rz', [])
                    
                    if len(rx) != len(ry) or len(ry) != len(rz):
                        print(f'Error: Length mismatch in {filename}')
                        continue
                    
                    task_info = data.get('task', {})
                    start_index = task_info.get('start', 0)
                    end_index = task_info.get('end', len(rx))
                    
                    coords = [(x, y, z) for i, (x, y, z) in enumerate(zip(rx, ry, rz))
                              if is_valid_point(x, y, z) and start_index <= i < end_index]
                    
                    if not coords:
                        print(f"No valid points found in range for {filename}.")
                        continue
                    
                    rx_filtered, ry_filtered, rz_filtered = zip(*coords)
                    
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(rx_filtered, ry_filtered, rz_filtered, s=10, alpha=0.5)
                    
                    # 添加 ados 信息作为批注
                    ados_info = f"ADOS info:\n{json.dumps(data['ados'], indent=2)}"
                    plt.figtext(0.02, 0.02, ados_info, fontsize=8, va="bottom", ha="left")
                    
                    ax.set_xlabel('RX')
                    ax.set_ylabel('RY')
                    ax.set_zlabel('RZ')
                    plt.title(f'Eye Gaze Data for {filename}')
                    
                    # 将图像保存到文件
                    output_filename = os.path.splitext(filename)[0] + '.png'
                    plt.savefig(os.path.join(output_folder_path, output_filename))
                    
                    plt.close(fig)  # 关闭图象，避免过多开销

            except json.JSONDecodeError as e:
                print(f'Invalid JSON in file: {filename}. Error: {e}')
            except KeyError as e:
                print(f'Missing key in file: {filename}. Error: {e}')

print('Search and plotting completed.')
