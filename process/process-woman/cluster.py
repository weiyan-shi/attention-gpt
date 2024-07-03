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

plt.switch_backend('TkAgg')
plt.ion()

PathCSV = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\woman\\merged_data.csv'
OutputCSV = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\woman\\ClusteringResults_new.csv'

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

def process_combination(row, df):
    participant = row['Participant']
    stimulus = row['Stimulus']
    trial = row['Trial']
    patient_type = row['Class']
    
    data_subset = df[(df['Participant'] == participant) & (df['Stimulus'] == stimulus ) & (df['Trial'] == trial) & (df['Class'] == patient_type) & (df['Category Right'] == 'Fixation')]
    
    X = data_subset['Point of Regard Right X [px]'].values
    Y = data_subset['Point of Regard Right Y [px]'].values
    
    points = np.column_stack((X, Y))

    try:
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
    
        print(f"Processed Participant: {participant}, Stimulus: {stimulus}, Type: {patient_type}")
        
        return {
            'Patient Type': patient_type,
            'Participant': participant,
            'Stimulus': trial+'-'+stimulus,
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
        }
    
    except Exception as e:
        print(f"Error processing Participant={participant}, Stimulus={stimulus}, Type={patient_type}: {e}")
        return None

df = pd.read_csv(PathCSV)

unique_combinations = df[['Participant', 'Stimulus', 'Trial', 'Class']].drop_duplicates()

if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()-6) as pool:
        results = pool.starmap(process_combination, [(row, df) for _, row in unique_combinations.iterrows()])

    results = [result for result in results if result is not None]
    results_df = pd.DataFrame(results)
    results_df.to_csv(OutputCSV, index=False)
