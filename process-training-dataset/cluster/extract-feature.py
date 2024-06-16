import os
import numpy as np
import pandas as pd
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter1d
import cv2

# 设置路径
PathASD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\TrainingData\\ASD\\'
PathTD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\TrainingData\\TD\\'
PathImage = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\TrainingData\\Images\\'
OutputCSV = 'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\ClusteringResults.csv'

files = os.listdir(PathImage)

def adjust_coordinates(x, y, max_x, max_y):
    x = np.clip(x - 1, 0, max_x - 1)
    y = np.clip(y - 1, 0, max_y - 1)
    return x, y

def find_optimal_kmeans_clusters(data, max_clusters=10):
    best_n_clusters = 1
    best_score = -1
    for n in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=0).fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n
    return best_n_clusters

def perform_kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    return labels

def calculate_k_distance(data, k=4):
    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    neighbors = nearest_neighbors.fit(data)
    distances, _ = neighbors.kneighbors(data)
    distances = np.sort(distances[:, k-1], axis=0)
    return distances

def find_elbow_point(distances):
    smoothed_distances = gaussian_filter1d(distances, sigma=2)
    second_derivative = np.diff(smoothed_distances, 2)
    elbow_point = np.argmax(second_derivative) + 2  # 加2是因为二阶差分减少了2个点
    return smoothed_distances[elbow_point]

def perform_dbscan_clustering(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = dbscan.labels_
    return labels

def perform_gmm_clustering(data, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(data)
    labels = gmm.predict(data)
    return labels

def perform_birch_clustering(data, threshold, branching_factor):
    birch = Birch(threshold=threshold, branching_factor=branching_factor).fit(data)
    labels = birch.predict(data)
    return labels

def find_optimal_dbscan_params(data):
    k_distances = calculate_k_distance(data)
    eps_value = find_elbow_point(k_distances)
    min_samples_value = max(1, int(len(data) * 0.01))
    return eps_value, min_samples_value

def find_optimal_birch_params(data, max_threshold=1.0, max_branching_factor=100):
    best_threshold = 0.5
    best_branching_factor = 50
    best_score = -1
    for threshold in np.linspace(0.1, max_threshold, 10):
        for branching_factor in range(20, max_branching_factor + 1, 10):
            birch = Birch(threshold=threshold, branching_factor=branching_factor).fit(data)
            labels = birch.labels_
            if len(set(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_branching_factor = branching_factor
    return best_threshold, best_branching_factor

def evaluate_clustering(data, labels):
    silhouette_avg = silhouette_score(data, labels)
    ch_score = calinski_harabasz_score(data, labels)
    db_score = davies_bouldin_score(data, labels)
    return silhouette_avg, ch_score, db_score

def noise_ratio(labels):
    noise_count = np.sum(labels == -1)
    total_count = len(labels)
    return noise_count / total_count

results = []

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

    # Adjust ASD coordinates and perform clustering
    try:
        X_ASD = dataASD[:, 1].astype(int)
        Y_ASD = dataASD[:, 2].astype(int)
        X_ASD, Y_ASD = adjust_coordinates(X_ASD, Y_ASD, ImgCol, ImgRow)
        asd_points = np.column_stack((X_ASD, Y_ASD))
        
        # KMeans clustering for ASD
        optimal_k_asd = find_optimal_kmeans_clusters(asd_points)
        kmeans_labels_asd = perform_kmeans_clustering(asd_points, n_clusters=optimal_k_asd)
        kmeans_scores_asd = evaluate_clustering(asd_points, kmeans_labels_asd)
        
        # DBSCAN clustering for ASD
        eps_asd, min_samples_asd = find_optimal_dbscan_params(asd_points)
        dbscan_labels_asd = perform_dbscan_clustering(asd_points, eps=eps_asd, min_samples=min_samples_asd)
        if len(set(dbscan_labels_asd)) > 1:
            dbscan_scores_asd = evaluate_clustering(asd_points, dbscan_labels_asd)
        else:
            dbscan_scores_asd = (None, None, None)
        dbscan_noise_ratio_asd = noise_ratio(dbscan_labels_asd)
        
        # GMM clustering for ASD
        optimal_gmm_asd = find_optimal_kmeans_clusters(asd_points)  # 使用与 KMeans 相同的方法找到最佳 GMM 组件数
        gmm_labels_asd = perform_gmm_clustering(asd_points, n_components=optimal_gmm_asd)
        gmm_scores_asd = evaluate_clustering(asd_points, gmm_labels_asd)
        
        # BIRCH clustering for ASD
        threshold_asd, branching_factor_asd = find_optimal_birch_params(asd_points)
        birch_labels_asd = perform_birch_clustering(asd_points, threshold=threshold_asd, branching_factor=branching_factor_asd)
        birch_scores_asd = evaluate_clustering(asd_points, birch_labels_asd)
        
        # Append ASD results
        results.append({
            'Patient Type': 'ASD',
            'Patient ID': FileName,
            'KMeans Silhouette Score': kmeans_scores_asd[0],
            'KMeans CH Score': kmeans_scores_asd[1],
            'KMeans DB Score': kmeans_scores_asd[2],
            'DBSCAN Silhouette Score': dbscan_scores_asd[0],
            'DBSCAN CH Score': dbscan_scores_asd[1],
            'DBSCAN DB Score': dbscan_scores_asd[2],
            'DBSCAN Noise Ratio': dbscan_noise_ratio_asd,
            'GMM Silhouette Score': gmm_scores_asd[0],
            'GMM CH Score': gmm_scores_asd[1],
            'GMM DB Score': gmm_scores_asd[2],
            'BIRCH Silhouette Score': birch_scores_asd[0],
            'BIRCH CH Score': birch_scores_asd[1],
            'BIRCH DB Score': birch_scores_asd[2]
        })
        
    except IndexError as e:
        print(f"IndexError for {FileName} in ASD data: {e}")
        continue
    
    # Adjust TD coordinates and perform clustering
    try:
        X_TD = dataTD[:, 1].astype(int)
        Y_TD = dataTD[:, 2].astype(int)
        X_TD, Y_TD = adjust_coordinates(X_TD, Y_TD, ImgCol, ImgRow)
        td_points = np.column_stack((X_TD, Y_TD))
        
        # KMeans clustering for TD
        optimal_k_td = find_optimal_kmeans_clusters(td_points)
        kmeans_labels_td = perform_kmeans_clustering(td_points, n_clusters=optimal_k_td)
        kmeans_scores_td = evaluate_clustering(td_points, kmeans_labels_td)
        
        # DBSCAN clustering for TD
        eps_td, min_samples_td = find_optimal_dbscan_params(td_points)
        dbscan_labels_td = perform_dbscan_clustering(td_points, eps=eps_td, min_samples=min_samples_td)
        if len(set(dbscan_labels_td)) > 1:
            dbscan_scores_td = evaluate_clustering(td_points, dbscan_labels_td)
        else:
            dbscan_scores_td = (None, None, None)
        dbscan_noise_ratio_td = noise_ratio(dbscan_labels_td)
        
        # GMM clustering for TD
        optimal_gmm_td = find_optimal_kmeans_clusters(td_points)  # 使用与 KMeans 相同的方法找到最佳 GMM 组件数
        gmm_labels_td = perform_gmm_clustering(td_points, n_components=optimal_gmm_td)
        gmm_scores_td = evaluate_clustering(td_points, gmm_labels_td)
        
        # BIRCH clustering for TD
        threshold_td, branching_factor_td = find_optimal_birch_params(td_points)
        birch_labels_td = perform_birch_clustering(td_points, threshold=threshold_td, branching_factor=branching_factor_td)
        birch_scores_td = evaluate_clustering(td_points, birch_labels_td)
        
        # Append TD results
        results.append({
            'Patient Type': 'TD',
            'Patient ID': FileName,
            'KMeans Silhouette Score': kmeans_scores_td[0],
            'KMeans CH Score': kmeans_scores_td[1],
            'KMeans DB Score': kmeans_scores_td[2],
            'DBSCAN Silhouette Score': dbscan_scores_td[0],
            'DBSCAN CH Score': dbscan_scores_td[1],
            'DBSCAN DB Score': dbscan_scores_td[2],
            'DBSCAN Noise Ratio': dbscan_noise_ratio_td,
            'GMM Silhouette Score': gmm_scores_td[0],
            'GMM CH Score': gmm_scores_td[1],
            'GMM DB Score': gmm_scores_td[2],
            'BIRCH Silhouette Score': birch_scores_td[0],
            'BIRCH CH Score': birch_scores_td[1],
            'BIRCH DB Score': birch_scores_td[2]
        })
        
    except IndexError as e:
        print(f"IndexError for {FileName} in TD data: {e}")
        continue

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OutputCSV, index=False)