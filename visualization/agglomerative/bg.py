import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import ConvexHull
import cv2
import os

# Label distribution for ASD in file 33:
# Label 0: 50 points (32.05%)
# Label 1: 25 points (16.03%)
# Label 2: 60 points (38.46%)
# Label 3: 21 points (13.46%)
# Label distribution for TD in file 33:
# Label 0: 13 points (11.82%)
# Label 1: 30 points (27.27%)
# Label 2: 42 points (38.18%)
# Label 3: 25 points (22.73%)
# Processed and saved clustered images for 33

# 设置路径
PathASD = "C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pics\\TrainingData\\ASD\\"
PathTD = "C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pics\\TrainingData\\TD\\"
PathImage = (
    "C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pics\\TrainingData\\Images\\"
)
OutputDir = "C:\\Users\\86178\\Desktop\\attention-gpt\\visualization\\agglomerative\\res"

files = ["33"]

n_clusters = 4  # Number of clusters for Agglomerative Clustering

# Define a list of vibrant colors for clustering
colors = {
    "Green": (0, 255, 0),       # Green: Correct in BGR
    "Orange": (0, 165, 255),    # Orange: Correct in BGR
    "Yellow": (0, 255, 255),    # Yellow: Correct in BGR
    "Pink": (203, 192, 255),    # Pink: Adjusted to (203, 192, 255) in BGR
    "Red": (0, 0, 255),         # Red: Correct in BGR
    "Blue": (255, 0, 0),        # Blue: Correct in BGR
    "White": (255, 255, 255),
}

def adjust_coordinates(x, y, max_x, max_y):
    x = np.clip(x - 1, 0, max_x - 1)
    y = np.clip(y - 1, 0, max_y - 1)
    return x, y

def perform_agglomerative_clustering(data, n_clusters):
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(data)
    labels = clustering.labels_
    return labels

def print_label_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    for label, count in zip(unique, counts):
        percentage = count / total * 100
        print(f"Label {label}: {count} points ({percentage:.2f}%)")

for file in files:
    FileName = os.path.splitext(file)[0]
    try:
        DataASD = pd.read_csv(f"{PathASD}ASD_scanpath_{FileName}.txt", delimiter=",")
        dataASD = DataASD.values

        DataTD = pd.read_csv(f"{PathTD}TD_scanpath_{FileName}.txt", delimiter=",")
        dataTD = DataTD.values
    except Exception as e:
        print(f"Error reading data for {FileName}: {e}")
        continue

    Img_ASD = cv2.imread(f"{PathImage}{FileName}.png")
    Img_TD = Img_ASD.copy()  # Copy for TD processing
    ImgRow, ImgCol, _ = Img_ASD.shape

    try:
        # Darken the image by 80%
        Img_ASD_darkened = cv2.addWeighted(Img_ASD, 0.2, np.zeros_like(Img_ASD), 0.8, 0)
        Img_TD_darkened = cv2.addWeighted(Img_TD, 0.2, np.zeros_like(Img_TD), 0.8, 0)

        # ASD Data Clustering
        X_ASD = dataASD[:, 1].astype(int)
        Y_ASD = dataASD[:, 2].astype(int)
        X_ASD, Y_ASD = adjust_coordinates(X_ASD, Y_ASD, ImgCol, ImgRow)
        points_ASD = np.column_stack((X_ASD, Y_ASD))

        # Apply Agglomerative Clustering
        agglomerative_labels_ASD = perform_agglomerative_clustering(points_ASD, n_clusters)

        # Print label distribution for ASD
        print(f"Label distribution for ASD in file {FileName}:")
        print_label_distribution(agglomerative_labels_ASD)

        # Create an empty image for the convex hulls
        overlay_ASD = np.zeros_like(Img_ASD)

        # Overlay clustering results with convex hulls
        for label in set(agglomerative_labels_ASD):
            label_points = points_ASD[agglomerative_labels_ASD == label]

            if len(label_points) >= 3:  # Convex hull requires at least 3 points
                hull = ConvexHull(label_points)
                hull_points = label_points[hull.vertices]

                # Draw and fill the convex hull
                color = colors['Green'] if label == 1 else colors['Yellow'] if label == 0 else colors['Pink'] if label == 2 else colors['Orange'] if label == 3 else colors['White']
                cv2.fillConvexPoly(overlay_ASD, hull_points, color)

        # Apply transparency to the filled hulls
        overlay_ASD_transparent = cv2.addWeighted(overlay_ASD, 0.8, np.zeros_like(overlay_ASD), 0.2, 0)

        # Combine the darkened image and the transparent hulls
        combined_ASD = cv2.addWeighted(Img_ASD_darkened, 1.0, overlay_ASD_transparent, 1.0, 0)

        # TD Data Clustering
        X_TD = dataTD[:, 1].astype(int)
        Y_TD = dataTD[:, 2].astype(int)
        X_TD, Y_TD = adjust_coordinates(X_TD, Y_TD, ImgCol, ImgRow)
        points_TD = np.column_stack((X_TD, Y_TD))

        # Apply Agglomerative Clustering
        agglomerative_labels_TD = perform_agglomerative_clustering(points_TD, n_clusters)

        # Print label distribution for TD
        print(f"Label distribution for TD in file {FileName}:")
        print_label_distribution(agglomerative_labels_TD)

        # Create an empty image for the convex hulls
        overlay_TD = np.zeros_like(Img_TD)

        # Overlay clustering results with convex hulls
        for label in set(agglomerative_labels_TD):
            label_points = points_TD[agglomerative_labels_TD == label]

            if len(label_points) >= 3:  # Convex hull requires at least 3 points
                hull = ConvexHull(label_points)
                hull_points = label_points[hull.vertices]

                # Draw and fill the convex hull
                color = colors['Green'] if label == 1 else colors['Yellow'] if label == 0 else colors['Pink'] if label == 2 else colors['Orange'] if label == 3 else colors['White']
                cv2.fillConvexPoly(overlay_TD, hull_points, color)

        # Apply transparency to the filled hulls
        overlay_TD_transparent = cv2.addWeighted(overlay_TD, 0.8, np.zeros_like(overlay_TD), 0.2, 0)

        # Combine the darkened image and the transparent hulls
        combined_TD = cv2.addWeighted(Img_TD_darkened, 1.0, overlay_TD_transparent, 1.0, 0)

        # Save the output images
        output_path_ASD = os.path.join(OutputDir, f"{FileName}_ASD_agglomerative.png")
        output_path_TD = os.path.join(OutputDir, f"{FileName}_TD_agglomerative.png")
        cv2.imwrite(output_path_ASD, combined_ASD)
        cv2.imwrite(output_path_TD, combined_TD)
        print(f"Processed and saved clustered images for {FileName}")

    except IndexError as e:
        print(f"IndexError for {FileName}: {e}")
        continue
