import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import random
import multiprocessing as mp

# Ensure the TkAgg backend is used
plt.switch_backend('TkAgg')
plt.ion()  # Turn on interactive mode

# 设置路径
PathCSV = 'C:\\Users\\86178\\Desktop\\attention-gpt\\woman\\merged_data.csv'
OutputCSV = 'C:\\Users\\86178\\Desktop\\attention-gpt\\woman\\ClusteringResults.csv'

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
        if len(np.unique(labels)) > 1:  # 确保至少有两个簇
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
    # Ensure eps_value is greater than 0.0
    if eps_value <= 0.0:
        eps_value = 0.1  # Set a default minimum value for eps
    min_samples_value = max(1, int(len(data) * 0.1))
    return eps_value, min_samples_value

def find_optimal_birch_params(data, max_threshold=1.0, max_branching_factor=100, n_iter=10):
    best_threshold = 0.5
    best_branching_factor = 50
    best_score = -1

    for _ in range(n_iter):
        threshold = random.uniform(0.1, max_threshold)
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

def evaluate_clustering(data, labels):
    if len(np.unique(labels)) > 1:  # 确保至少有两个簇
        silhouette_avg = silhouette_score(data, labels)
        ch_score = calinski_harabasz_score(data, labels)
        db_score = davies_bouldin_score(data, labels)
        return silhouette_avg, ch_score, db_score
    else:
        return None, None, None

def noise_ratio(labels):
    noise_count = np.sum(labels == -1)
    total_count = len(labels)
    return noise_count / total_count

def process_combination(row, df):
    participant = row['Participant']
    stimulus = row['Stimulus']
    trial = row['Trial']
    patient_type = row['Class']
    
    data_subset = df[(df['Participant'] == participant) & (df['Stimulus'] == stimulus ) & (df['Trial'] == trial) & (df['Class'] == patient_type) & (df['Category Right'] == 'Fixation')]
    
    X = data_subset['Point of Regard Right X [px]'].values
    Y = data_subset['Point of Regard Right Y [px]'].values
    
    points = np.column_stack((X, Y))
    # points = points[::10]

    try:
        # KMeans clustering
        optimal_k = find_optimal_kmeans_clusters(points)
        kmeans_labels = perform_kmeans_clustering(points, n_clusters=optimal_k)
        kmeans_scores = evaluate_clustering(points, kmeans_labels)
        
        # DBSCAN clustering
        eps, min_samples = find_optimal_dbscan_params(points)
        dbscan_labels = perform_dbscan_clustering(points, eps=eps, min_samples=min_samples)
        if len(set(dbscan_labels)) > 1:
            dbscan_scores = evaluate_clustering(points, dbscan_labels)
        else:
            dbscan_scores = (None, None, None)
        dbscan_noise_ratio = noise_ratio(dbscan_labels)
        
        # GMM clustering
        optimal_gmm = find_optimal_kmeans_clusters(points)
        gmm_labels = perform_gmm_clustering(points, n_components=optimal_gmm)
        gmm_scores = evaluate_clustering(points, gmm_labels)
        
        # BIRCH clustering
        threshold, branching_factor = find_optimal_birch_params(points)
        birch_labels = perform_birch_clustering(points, threshold=threshold, branching_factor=branching_factor)
        birch_scores = evaluate_clustering(points, birch_labels)

        print(f"Processed Participant: {participant}, Stimulus: {stimulus}, Type: {patient_type}")
        
        return {
            'Patient Type': patient_type,
            'Participant': participant,
            'Stimulus': stimulus,
            'Trial' : trial,
            'KMeans Silhouette Score': kmeans_scores[0],
            'KMeans CH Score': kmeans_scores[1],
            'KMeans DB Score': kmeans_scores[2],
            'DBSCAN Silhouette Score': dbscan_scores[0],
            'DBSCAN CH Score': dbscan_scores[1],
            'DBSCAN DB Score': dbscan_scores[2],
            'DBSCAN Noise Ratio': dbscan_noise_ratio,
            'GMM Silhouette Score': gmm_scores[0],
            'GMM CH Score': gmm_scores[1],
            'GMM DB Score': gmm_scores[2],
            'BIRCH Silhouette Score': birch_scores[0],
            'BIRCH CH Score': birch_scores[1],
            'BIRCH DB Score': birch_scores[2]
        }
    
    except Exception as e:
        print(f"Error processing Participant={participant}, Stimulus={stimulus}, Type={patient_type}: {e}")
        return None

# 读取CSV文件
df = pd.read_csv(PathCSV)

# 获取唯一的Participant和Movie组合
unique_combinations = df[['Participant', 'Stimulus', 'Trial', 'Class']].drop_duplicates()

# Use multiprocessing to process the combinations
if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()-6) as pool:
        results = pool.starmap(process_combination, [(row, df) for _, row in unique_combinations.iterrows()])

    # Filter out None results
    results = [result for result in results if result is not None]
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OutputCSV, index=False)
