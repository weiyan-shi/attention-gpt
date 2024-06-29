import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cv2

# Ensure the TkAgg backend is used
plt.switch_backend('TkAgg')
plt.ion()  # Turn on interactive mode

# 设置路径
PathASD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\TrainingData\\ASD\\'
PathTD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\TrainingData\\TD\\'
PathImage = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\TrainingData\\Images\\'
PathScatterASD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\AdditionalData\\asd-scatter\\'
PathScatterTD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\AdditionalData\\td-scatter\\'

if not os.path.exists(PathScatterASD):
    os.makedirs(PathScatterASD)
if not os.path.exists(PathScatterTD):
    os.makedirs(PathScatterTD)

files = os.listdir(PathImage)

def adjust_coordinates(x, y, max_x, max_y):
    x = np.clip(x - 1, 0, max_x - 1)
    y = np.clip(y - 1, 0, max_y - 1)
    return x, y

def calculate_time_points(durations):
    time_points = np.cumsum(durations)
    return time_points

def plot_time_based_scatter(points, times, title, save_path):
    norm = Normalize(vmin=np.min(times), vmax=np.max(times))
    cmap = plt.get_cmap('viridis')
    plt.figure()
    sc = plt.scatter(points[:, 0], points[:, 1], c=times, cmap=cmap, s=10, edgecolors='none', alpha=0.6)
    plt.colorbar(sc, label='Time')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y axis to match image coordinate system
    plt.savefig(save_path)
    plt.close()

for file in files:
    if file in ['.', '..']:
        continue
    else:
        FileName = os.path.splitext(file)[0]
        
        try:
            DataASD = pd.read_csv(f'{PathASD}ASD_scanpath_{FileName}.txt', delimiter=',')
            dataASD = DataASD.values
            print(f'ASD data for {FileName}:')
            print(DataASD.head())

            DataTD = pd.read_csv(f'{PathTD}TD_scanpath_{FileName}.txt', delimiter=',')
            dataTD = DataTD.values
            print(f'TD data for {FileName}:')
            print(DataTD.head())
        except Exception as e:
            print(f"Error reading data for {FileName}: {e}")
            continue

    Img = cv2.imread(f'{PathImage}{FileName}.png')
    ImgRow, ImgCol, _ = Img.shape

    # Plot ASD time-based scatter plot
    try:
        X_ASD = dataASD[:, 1].astype(int)
        Y_ASD = dataASD[:, 2].astype(int)
        Duration_ASD = dataASD[:, 3].astype(int)
        Time_ASD = calculate_time_points(Duration_ASD)
        X_ASD, Y_ASD = adjust_coordinates(X_ASD, Y_ASD, ImgCol, ImgRow)
        asd_points = np.column_stack((X_ASD, Y_ASD))
        
        # Plot and save time-based scatter plot
        plot_time_based_scatter(asd_points, Time_ASD, f'ASD Time-based Scatter - {FileName}', f'{PathScatterASD}{FileName}_scatter.png')
        
    except IndexError as e:
        print(f"IndexError for {FileName} in ASD data: {e}")
        continue

    # Plot TD time-based scatter plot
    try:
        X_TD = dataTD[:, 1].astype(int)
        Y_TD = dataTD[:, 2].astype(int)
        Duration_TD = dataTD[:, 3].astype(int)
        Time_TD = calculate_time_points(Duration_TD)
        X_TD, Y_TD = adjust_coordinates(X_TD, Y_TD, ImgCol, ImgRow)
        td_points = np.column_stack((X_TD, Y_TD))
        
        # Plot and save time-based scatter plot
        plot_time_based_scatter(td_points, Time_TD, f'TD Time-based Scatter - {FileName}', f'{PathScatterTD}{FileName}_scatter.png')
        
    except IndexError as e:
        print(f"IndexError for {FileName} in TD data: {e}")
        continue
    break
