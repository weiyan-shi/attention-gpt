import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter1d
import cv2
import os
import time


# 设置路径
PathASD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pics\\TrainingData\\ASD\\'
PathTD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pics\\TrainingData\\TD\\'
PathImage = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pics\\TrainingData\\Images\\'
OutputDir = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pics\\OutputImages\\'

files = ['3']

def adjust_coordinates(x, y, max_x, max_y):
    x = np.clip(x - 1, 0, max_x - 1)
    y = np.clip(y - 1, 0, max_y - 1)
    return x, y

def calculate_k_distance(data, k=4):
    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    neighbors = nearest_neighbors.fit(data)
    distances, _ = neighbors.kneighbors(data)
    distances = np.sort(distances[:, k-1], axis=0)
    return distances

def find_elbow_point(distances):
    smoothed_distances = gaussian_filter1d(distances, sigma=2)
    second_derivative = np.diff(smoothed_distances, 2)
    elbow_point = np.argmax(second_derivative) + 2
    return smoothed_distances[elbow_point]

def find_optimal_dbscan_params(data):
    k_distances = calculate_k_distance(data)
    eps_value = find_elbow_point(k_distances)
    if eps_value <= 0.0:
        eps_value = 0.1
    min_samples_value = max(1, int(len(data) * 0.1))
    return eps_value, min_samples_value

def perform_dbscan_clustering(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = dbscan.labels_
    return labels

for file in files:
    FileName = os.path.splitext(file)[0]
    try:
        DataASD = pd.read_csv(f'{PathASD}ASD_scanpath_{FileName}.txt', delimiter=',')
        dataASD = DataASD.values

        DataTD = pd.read_csv(f'{PathTD}TD_scanpath_{FileName}.txt', delimiter=',')
        dataTD = DataTD.values
    except Exception as e:
        print(f"Error reading data for {FileName}: {e}")
        continue

    Img_ASD = cv2.imread(f'{PathImage}{FileName}.png')
    Img_TD = Img_ASD.copy()  # Copy for TD processing
    ImgRow, ImgCol, _ = Img_ASD.shape

    try:
        # ASD Data Clustering
        X_ASD = dataASD[:, 1].astype(int)
        Y_ASD = dataASD[:, 2].astype(int)
        X_ASD, Y_ASD = adjust_coordinates(X_ASD, Y_ASD, ImgCol, ImgRow)
        points_ASD = np.column_stack((X_ASD, Y_ASD))

        eps, min_samples = find_optimal_dbscan_params(points_ASD)
        dbscan_labels_ASD = perform_dbscan_clustering(points_ASD, eps=50, min_samples=10)

        # Overlay clustering results on ASD image
        for i, label in enumerate(dbscan_labels_ASD):
            if label == -1:  # Noise
                color = (0, 0, 0)  # Black
            else:
                color = (0, 255, 0) if label == 0 else (0, 0, 255)
            cv2.circle(Img_ASD, (X_ASD[i], Y_ASD[i]), 5, color, -1)

        # TD Data Clustering
        X_TD = dataTD[:, 1].astype(int)
        Y_TD = dataTD[:, 2].astype(int)
        X_TD, Y_TD = adjust_coordinates(X_TD, Y_TD, ImgCol, ImgRow)
        points_TD = np.column_stack((X_TD, Y_TD))

        eps, min_samples = find_optimal_dbscan_params(points_TD)
        dbscan_labels_TD = perform_dbscan_clustering(points_TD, eps=50, min_samples=10)

        # Overlay clustering results on TD image
        for i, label in enumerate(dbscan_labels_TD):
            if label == -1:  # Noise
                color = (0, 0, 0)  # Black
            else:
                color = (255, 0, 0) if label == 0 else (0, 255, 255)
            cv2.circle(Img_TD, (X_TD[i], Y_TD[i]), 5, color, -1)

        # Save the output images
        output_path_ASD = os.path.join(OutputDir, f'{FileName}_ASD_clustered.png')
        output_path_TD = os.path.join(OutputDir, f'{FileName}_TD_clustered.png')
        cv2.imwrite(output_path_ASD, Img_ASD)
        cv2.imwrite(output_path_TD, Img_TD)
        print(f"Processed and saved clustered images for {FileName}")

    except IndexError as e:
        print(f"IndexError for {FileName}: {e}")
        continue
