import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Ensure the TkAgg backend is used
plt.switch_backend('TkAgg')
plt.ion()  # Turn on interactive mode

# 设置路径
PathASD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\TrainingData\\ASD\\'
PathTD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\TrainingData\\TD\\'
PathImage = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\TrainingData\\Images\\'
PathFixPtsASD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\AdditionalData\\asd-fixpts-new\\'
PathFixMapsASD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\TrainingData\\asd-fixmaps\\'
PathHeatMapsASD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\AdditionalData\\asd-heatmaps\\'
PathFixPtsTD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\AdditionalData\\td-fixpts-new\\'
PathFixMapsTD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\TrainingData\\td-fixmaps\\'
PathHeatMapsTD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\AdditionalData\\td-heatmaps\\'

files = os.listdir(PathImage)

for file in files:
    if file in ['.', '..']:
        continue
    else:
        FileName = os.path.splitext(file)[0]
        
        # Try reading the data and inspect the structure
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

    # Adjusting coordinates to be within the image bounds
    def adjust_coordinates(x, y, max_x, max_y):
        x = np.clip(x - 1, 0, max_x - 1)
        y = np.clip(y - 1, 0, max_y - 1)
        return x, y

    # ASD
    # fixation map initialization
    FixationPoints = np.zeros((ImgRow, ImgCol))

    try:
        X_ASD = dataASD[:, 1].astype(int)
        Y_ASD = dataASD[:, 2].astype(int)
        X_ASD, Y_ASD = adjust_coordinates(X_ASD, Y_ASD, ImgCol, ImgRow)
    except IndexError as e:
        print(f"IndexError for {FileName} in ASD data: {e}")
        continue

    FixationPoints[Y_ASD, X_ASD] = 1

    # write fixation points
    cv2.imwrite(f'{PathFixPtsASD}{FileName}_f.png', FixationPoints * 255)

    # write fixation map
    window = gaussian_filter(FixationPoints, sigma=43)
    FixationMap = cv2.normalize(window, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f'{PathFixMapsASD}{FileName}_s.png', FixationMap)

    # heat map
    plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
    plt.imshow(FixationMap, cmap='hot', alpha=0.4)
    plt.colorbar()
    plt.savefig(f'{PathHeatMapsASD}{FileName}_h.png')
    plt.close()

    # TD
    # fixation map initialization
    FixationPoints = np.zeros((ImgRow, ImgCol))

    try:
        X_TD = dataTD[:, 1].astype(int)
        Y_TD = dataTD[:, 2].astype(int)
        X_TD, Y_TD = adjust_coordinates(X_TD, Y_TD, ImgCol, ImgRow)
    except IndexError as e:
        print(f"IndexError for {FileName} in TD data: {e}")
        continue

    FixationPoints[Y_TD, X_TD] = 1

    # write fixation points
    cv2.imwrite(f'{PathFixPtsTD}{FileName}_f.png', FixationPoints * 255)

    # write fixation map
    window = gaussian_filter(FixationPoints, sigma=43)
    FixationMap = cv2.normalize(window, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f'{PathFixMapsTD}{FileName}_s.png', FixationMap)

    # heat map
    plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
    plt.imshow(FixationMap, cmap='hot', alpha=0.4)
    plt.colorbar()
    plt.savefig(f'{PathHeatMapsTD}{FileName}_h.png')
    plt.close()
