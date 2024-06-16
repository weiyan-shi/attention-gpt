import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Ensure the TkAgg backend is used
plt.switch_backend('TkAgg')
plt.ion()  # Turn on interactive mode

def is_valid_point(x, y, z):
    return x is not None and y is not None and z is not None

def merge_gaze_points(coords, avg_distance):
    distance_matrix = squareform(pdist(coords))
    merged_points = []
    visited = set()

    for i in range(len(coords)):
        if i in visited:
            continue

        group = [coords[i]]
        visited.add(i)

        for j in range(i + 1, len(coords)):
            if j in visited:
                continue

            if distance_matrix[i][j] < avg_distance:
                group.append(coords[j])
                visited.add(j)

        if group:
            merged_x = np.mean([p[0] for p in group])
            merged_y = np.mean([p[1] for p in group])
            merged_z = np.mean([p[2] for p in group])
            merged_points.append((merged_x, merged_y, merged_z))

    return merged_points

folder_path = 'C:\\Users\\86178\\Desktop\\attention-gpt\\DREAMdataset'
output_folder_path = 'C:\\Users\\weiyan.shi\\Desktop\\attention-gpt\\output'

for root, dirs, files in os.walk(folder_path):
    for subdir in dirs:
        input_subfolder_path = os.path.join(folder_path, subdir)
        output_subfolder_path = os.path.join(output_folder_path, subdir)
        # if not os.path.exists(output_subfolder_path):
        #     os.makedirs(output_subfolder_path)
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

                                # 计算平均距离
                                avg_distance = 0.3*np.mean(pdist(coords))

                                # 合并距离小于平均距离的点
                                merged_coords = merge_gaze_points(coords, avg_distance)
                                rx_all, ry_all, rz_all = zip(*merged_coords)

                                desired_size = 2
                                desired_alpha = 0.7

                                rx_all = np.array(rx_all)
                                ry_all = np.array(ry_all)
                                rz_all = np.array(rz_all)

                                fig = plt.figure()
                                ax = fig.add_subplot(111, projection='3d')

                                ax.scatter(
                                    rx_all,
                                    ry_all,
                                    rz_all,
                                    s=desired_size,
                                    alpha=desired_alpha,
                                )

                                ax.plot(rx_all, ry_all, rz_all, linestyle='-', linewidth=1, alpha=0.5)

                                ax.set_xlabel('RX')
                                ax.set_ylabel('RY')
                                ax.set_zlabel('RZ')

                                ados_items = list(data['ados']['preTest'].items())
                                ados_info = f"ADOS info:{ados_items[7]}"
                                print(ados_info)

                                output_filename = os.path.splitext(filename)[0] + '_3D.png'
                                # plt.savefig(os.path.join(output_subfolder_path, output_filename))
                                plt.show()  # Display the plot

                    except json.JSONDecodeError as e:
                        print(f'Invalid JSON in file: {filename}. Error: {e}')
                    except KeyError as e:
                        print(f'Missing key in file: {filename}. Error: {e}')

print('Search and plotting completed.')
