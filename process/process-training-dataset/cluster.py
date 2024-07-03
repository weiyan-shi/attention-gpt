import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch, KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import cv2
import os

# 设置路径
PathASD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\TrainingDataset\\TrainingData\\ASD\\'
PathTD = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\TrainingDataset\\TrainingData\\TD\\'
PathImage = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\TrainingDataset\\TrainingData\\Images\\'
OutputCSV = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\TrainingDataset\\ClusteringResults_new.csv'

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
        if len(np.unique(labels)) > 1:
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n
    return best_n_clusters

def find_optimal_kmedoids_clusters(data, max_clusters=10):
    best_n_clusters = 1
    best_score = -1
    for n in range(2, max_clusters + 1):
        kmedoids = KMedoids(n_clusters=n, random_state=0).fit(data)
        labels = kmedoids.labels_
        if len(np.unique(labels)) > 1:
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n
    return best_n_clusters

def find_optimal_agglomerative_clusters(data, max_clusters=10):
    best_n_clusters = 2
    best_score = -1
    
    for n in range(2, max_clusters + 1):
        agglomerative = AgglomerativeClustering(n_clusters=n)
        labels = agglomerative.fit_predict(data)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n
    return best_n_clusters

def find_optimal_dbscan_params(data):
    k_distances = calculate_k_distance(data)
    eps_value = find_elbow_point(k_distances)
    if eps_value <= 0.0:
        eps_value = 0.1
    min_samples_value = max(1, int(len(data) * 0.1))
    return eps_value, min_samples_value

def find_optimal_optics_params(data):
    k_distances = calculate_k_distance(data)
    eps_value = find_elbow_point(k_distances)
    if eps_value <= 0.0:
        eps_value = 0.1
    min_samples_value = max(1, int(len(data) * 0.1))
    return eps_value, min_samples_value

def find_optimal_birch_params(data, max_threshold=1.0, max_branching_factor=100, n_iter=20):
    best_threshold = 0.01
    best_branching_factor = 50
    best_score = -1

    for _ in range(n_iter):
        threshold = random.uniform(0.001, max_threshold)
        branching_factor = random.randint(20, max_branching_factor)
        try:
            birch = Birch(threshold=threshold, branching_factor=branching_factor).fit(data)
            labels = birch.labels_
            if len(set(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_branching_factor = branching_factor
        except Exception as e:
            print(f"Error with threshold={threshold} and branching_factor={branching_factor}: {e}")
            continue

    if best_score == -1:
        print("Unable to find optimal BIRCH parameters. Please check your data and parameter ranges.")
    else:
        print(f"Best threshold: {best_threshold}, Best branching factor: {best_branching_factor}, Best score: {best_score}")

    return best_threshold, best_branching_factor

def find_optimal_gmm_clusters(data, max_clusters=10):
    best_n_clusters = 1
    best_score = -1
    for n in range(2, max_clusters + 1):
        gmm = GaussianMixture(n_components=n, random_state=0).fit(data)
        labels = gmm.predict(data)
        if len(np.unique(labels)) > 1:
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
    elbow_point = np.argmax(second_derivative) + 2
    return smoothed_distances[elbow_point]

def perform_dbscan_clustering(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = dbscan.labels_
    return labels

def perform_gmm_clustering(data, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(data)
    labels = gmm.predict(data)
    return labels

def perform_birch_clustering(data, threshold, branching_factor, min_threshold=0.001):
    while threshold > min_threshold:
        birch = Birch(threshold=threshold, branching_factor=branching_factor).fit(data)
        labels = birch.predict(data)
        unique_labels = np.unique(labels)
        if len(unique_labels) >= 3:
            return labels
        else:
            threshold /= 2
    raise ValueError(f"BIRCH clustering produced less than 3 clusters even with minimum threshold {min_threshold}")

def evaluate_clustering(data, labels):
    if len(np.unique(labels)) > 1:
        silhouette_avg = silhouette_score(data, labels)
        ch_score = calinski_harabasz_score(data, labels)
        db_score = davies_bouldin_score(data, labels)
        return silhouette_avg, ch_score, db_score
    else:
        return None, None, None
    
def perform_agglomerative_clustering(data, n_clusters):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters).fit(data)
    labels = agglomerative.labels_
    return labels

def perform_kmedoids_clustering(data, n_clusters):
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmedoids.labels_
    return labels

def perform_optics_clustering(data, eps, min_samples):
    optics = OPTICS(max_eps=eps, min_samples=min_samples).fit(data)
    labels = optics.labels_
    return labels

results = []

for file in files:
    if file in ['.', '..']:
        continue
    else:
        FileName = os.path.splitext(file)[0]
        try:
            DataASD = pd.read_csv(f'{PathASD}ASD_scanpath_{FileName}.txt', delimiter=',')
            dataASD = DataASD.values

            DataTD = pd.read_csv(f'{PathTD}TD_scanpath_{FileName}.txt', delimiter=',')
            dataTD = DataTD.values
        except Exception as e:
            print(f"Error reading data for {FileName}: {e}")
            continue

    Img = cv2.imread(f'{PathImage}{FileName}.png')
    ImgRow, ImgCol, _ = Img.shape

    try:
        X_ASD = dataASD[:, 1].astype(int)
        Y_ASD = dataASD[:, 2].astype(int)
        X_ASD, Y_ASD = adjust_coordinates(X_ASD, Y_ASD, ImgCol, ImgRow)
        points = np.column_stack((X_ASD, Y_ASD))

        optimal_k = find_optimal_kmeans_clusters(points)
        kmeans_labels = perform_kmeans_clustering(points, n_clusters=optimal_k)
        kmeans_scores = evaluate_clustering(points, kmeans_labels)

        optimal_k = find_optimal_kmedoids_clusters(points)
        kmedoids_labels = perform_kmedoids_clustering(points, n_clusters=optimal_k)
        kmedoids_scores = evaluate_clustering(points, kmedoids_labels)

        optimal_k = find_optimal_agglomerative_clusters(points, max_clusters=10)
        agglomerative_labels = perform_agglomerative_clustering(points, n_clusters=optimal_k)
        agglomerative_scores = evaluate_clustering(points, agglomerative_labels)

        threshold, branching_factor = find_optimal_birch_params(points)
        birch_labels = perform_birch_clustering(points, threshold=threshold, branching_factor=branching_factor)
        birch_scores = evaluate_clustering(points, birch_labels)

        eps, min_samples = find_optimal_dbscan_params(points)
        dbscan_labels = perform_dbscan_clustering(points, eps=eps, min_samples=min_samples)
        if len(set(dbscan_labels)) > 1:
            dbscan_scores = evaluate_clustering(points, dbscan_labels)
        else:
            dbscan_scores = (None, None, None)

        eps, min_samples = find_optimal_optics_params(points)
        optics_labels = perform_optics_clustering(points, eps=eps, min_samples=min_samples)
        if len(set(optics_labels)) > 1:
            optics_scores = evaluate_clustering(points, optics_labels)
        else:
            optics_scores = (None, None, None)

        optimal_gmm = find_optimal_gmm_clusters(points)
        gmm_labels = perform_gmm_clustering(points, n_components=optimal_gmm)
        gmm_scores = evaluate_clustering(points, gmm_labels)
        
        # Append ASD results
        results.append({
            'Patient Type': 'ASD',
            'Patient ID': FileName,
            'KMeans Silhouette Score': kmeans_scores[0],
            'KMeans CH Score': kmeans_scores[1],
            'KMeans DB Score': kmeans_scores[2],
            'DBSCAN Silhouette Score': dbscan_scores[0],
            'DBSCAN CH Score': dbscan_scores[1],
            'DBSCAN DB Score': dbscan_scores[2],
            'GMM Silhouette Score': gmm_scores[0],
            'GMM CH Score': gmm_scores[1],
            'GMM DB Score': gmm_scores[2],
            'BIRCH Silhouette Score': birch_scores[0],
            'BIRCH CH Score': birch_scores[1],
            'BIRCH DB Score': birch_scores[2],
            'Agglomerative Silhouette Score': agglomerative_scores[0],
            'Agglomerative CH Score': agglomerative_scores[1],
            'Agglomerative DB Score': agglomerative_scores[2],
            'KMedoids Silhouette Score': kmedoids_scores[0],
            'KMedoids CH Score': kmedoids_scores[1],
            'KMedoids DB Score': kmedoids_scores[2],
            'OPTICS Silhouette Score': optics_scores[0],
            'OPTICS CH Score': optics_scores[1],
            'OPTICS DB Score': optics_scores[2],
        })
        
    except IndexError as e:
        print(f"IndexError for {FileName} in ASD data: {e}")
        continue
    
    # Adjust TD coordinates and perform clustering
    try:
        X_TD = dataTD[:, 1].astype(int)
        Y_TD = dataTD[:, 2].astype(int)
        X_TD, Y_TD = adjust_coordinates(X_TD, Y_TD, ImgCol, ImgRow)
        points = np.column_stack((X_TD, Y_TD))
        
        optimal_k = find_optimal_kmeans_clusters(points)
        kmeans_labels = perform_kmeans_clustering(points, n_clusters=optimal_k)
        kmeans_scores = evaluate_clustering(points, kmeans_labels)

        optimal_k = find_optimal_kmedoids_clusters(points)
        kmedoids_labels = perform_kmedoids_clustering(points, n_clusters=optimal_k)
        kmedoids_scores = evaluate_clustering(points, kmedoids_labels)

        optimal_k = find_optimal_agglomerative_clusters(points, max_clusters=10)
        agglomerative_labels = perform_agglomerative_clustering(points, n_clusters=optimal_k)
        agglomerative_scores = evaluate_clustering(points, agglomerative_labels)

        threshold, branching_factor = find_optimal_birch_params(points)
        birch_labels = perform_birch_clustering(points, threshold=threshold, branching_factor=branching_factor)
        birch_scores = evaluate_clustering(points, birch_labels)

        eps, min_samples = find_optimal_dbscan_params(points)
        dbscan_labels = perform_dbscan_clustering(points, eps=eps, min_samples=min_samples)
        if len(set(dbscan_labels)) > 1:
            dbscan_scores = evaluate_clustering(points, dbscan_labels)
        else:
            dbscan_scores = (None, None, None)

        eps, min_samples = find_optimal_optics_params(points)
        optics_labels = perform_optics_clustering(points, eps=eps, min_samples=min_samples)
        if len(set(optics_labels)) > 1:
            optics_scores = evaluate_clustering(points, optics_labels)
        else:
            optics_scores = (None, None, None)

        optimal_gmm = find_optimal_gmm_clusters(points)
        gmm_labels = perform_gmm_clustering(points, n_components=optimal_gmm)
        gmm_scores = evaluate_clustering(points, gmm_labels)
        
        # Append ASD results
        results.append({
            'Patient Type': 'TD',
            'Patient ID': FileName,
            'KMeans Silhouette Score': kmeans_scores[0],
            'KMeans CH Score': kmeans_scores[1],
            'KMeans DB Score': kmeans_scores[2],
            'DBSCAN Silhouette Score': dbscan_scores[0],
            'DBSCAN CH Score': dbscan_scores[1],
            'DBSCAN DB Score': dbscan_scores[2],
            'GMM Silhouette Score': gmm_scores[0],
            'GMM CH Score': gmm_scores[1],
            'GMM DB Score': gmm_scores[2],
            'BIRCH Silhouette Score': birch_scores[0],
            'BIRCH CH Score': birch_scores[1],
            'BIRCH DB Score': birch_scores[2],
            'Agglomerative Silhouette Score': agglomerative_scores[0],
            'Agglomerative CH Score': agglomerative_scores[1],
            'Agglomerative DB Score': agglomerative_scores[2],
            'KMedoids Silhouette Score': kmedoids_scores[0],
            'KMedoids CH Score': kmedoids_scores[1],
            'KMedoids DB Score': kmedoids_scores[2],
            'OPTICS Silhouette Score': optics_scores[0],
            'OPTICS CH Score': optics_scores[1],
            'OPTICS DB Score': optics_scores[2],
        })

        print(f"Processed {FileName}")

        
    except IndexError as e:
        print(f"IndexError for {FileName} in TD data: {e}")
        continue

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OutputCSV, index=False)