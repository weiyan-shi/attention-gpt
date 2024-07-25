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
import time
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.stats import pointbiserialr

def CH(X, labels):
    return calinski_harabasz_score(X, labels)

def CSL(X, labels):
    unique_labels = np.unique(labels)
    intra_dists = np.sum([np.mean(np.linalg.norm(X[labels == i] - X[labels == i].mean(axis=0), axis=1)) 
                          for i in unique_labels if np.any(labels == i)])
    cluster_means = np.array([X[labels == i].mean(axis=0) for i in unique_labels if np.any(labels == i)])
    inter_dists = np.mean(np.linalg.norm(cluster_means - X.mean(axis=0), axis=1))
    return inter_dists / intra_dists

def DI(X, labels):
    unique_labels = np.unique(labels)
    clusters = [X[labels == i] for i in unique_labels if np.any(labels == i)]
    if len(clusters) < 2:
        return np.inf
    min_intercluster_dist = np.min([cdist(c1, c2).min() for i, c1 in enumerate(clusters) for c2 in clusters[i + 1:]])
    max_intracluster_dist = np.max([cdist(c, c).max() for c in clusters])
    return min_intercluster_dist / max_intracluster_dist

def DB(X, labels):
    return davies_bouldin_score(X, labels)

def DB_star(X, labels):
    unique_labels = np.unique(labels)
    clusters = [X[labels == i] for i in unique_labels if np.any(labels == i)]
    if len(clusters) < 2:
        return np.inf
    centroids = np.array([c.mean(axis=0) for c in clusters])
    R = np.zeros((len(clusters), len(clusters)))
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i != j:
                Si = np.mean(np.linalg.norm(clusters[i] - centroids[i], axis=1))
                Sj = np.mean(np.linalg.norm(clusters[j] - centroids[j], axis=1))
                Mij = np.linalg.norm(centroids[i] - centroids[j])
                R[i, j] = (Si + Sj) / Mij
    return np.mean(np.max(R, axis=1))

def GD33(X, labels):
    unique_labels = np.unique(labels)
    clusters = [X[labels == i] for i in unique_labels if np.any(labels == i)]
    if len(clusters) < 2:
        return np.inf
    min_intercluster_dist = np.min([cdist(c1, c2).min() for i, c1 in enumerate(clusters) for c2 in clusters[i + 1:]])
    max_intracluster_dist = np.max([cdist(c, c).mean() for c in clusters])
    return min_intercluster_dist / max_intracluster_dist

def PB(X, labels):
    labels = labels.ravel()
    dists = pairwise_distances(X)
    n = len(labels)
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri_dists = dists[upper_tri_indices]
    upper_tri_labels = (labels[upper_tri_indices[0]] == labels[upper_tri_indices[1]]).astype(int)
    pbc = pointbiserialr(upper_tri_labels, upper_tri_dists)
    return pbc.correlation

def PBM(X, labels):
    unique_labels = np.unique(labels)
    clusters = [X[labels == i] for i in unique_labels if np.any(labels == i)]
    if len(clusters) < 2:
        return np.inf
    centroid = X.mean(axis=0)
    cluster_centroids = np.array([c.mean(axis=0) for c in clusters])
    E1 = np.sum(np.linalg.norm(X - centroid, axis=1))
    Ek = np.sum([np.sum(np.linalg.norm(c - cluster_centroids[i], axis=1)) for i, c in enumerate(clusters)])
    Dk = np.max([np.linalg.norm(ci - cj) for i, ci in enumerate(cluster_centroids) for cj in cluster_centroids[i + 1:]])
    return (1 / len(clusters)) * (E1 / Ek) * Dk

def SC(X, labels):
    return silhouette_score(X, labels)

def STR(X, labels):
    dists = pairwise_distances(X)
    inter_cluster_dist = np.mean([np.min(dists[i, labels != labels[i]]) for i in range(len(labels))])
    intra_cluster_dist = np.mean([np.max(dists[i, labels == labels[i]]) for i in range(len(labels))])
    return inter_cluster_dist / intra_cluster_dist

plt.switch_backend('TkAgg')
plt.ion()

PathCSV = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\woman\\merged_data.csv'
OutputCSV = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\woman\\ClusteringResults_10.csv'

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
        return {
            'SC': silhouette_avg,
            'CH': ch_score,
            'DB': db_score,
            'CSL': CSL(data, labels),
            'DI': DI(data, labels),
            'DB*': DB_star(data, labels),
            'GD33': GD33(data, labels),
            'PB': PB(data, labels),
            'PBM': PBM(data, labels),
            'STR': STR(data, labels)
        }
    else:
        return {key: None for key in ['SC', 'CH', 'DB', 'CSL', 'DI', 'DB*', 'GD33', 'PB', 'PBM', 'STR']}

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
    
    if len(X) == 0 or len(Y) == 0:
        print(f"Skipping Participant={participant}, Stimulus={stimulus}, Type={patient_type} due to insufficient data points.")
        return None
    
    points = np.column_stack((X, Y))
    results = {}

    try:
        start_time = time.time()
        optimal_k = find_optimal_kmeans_clusters(points)
        kmeans_labels = perform_kmeans_clustering(points, n_clusters=optimal_k)
        kmeans_scores = evaluate_clustering(points, kmeans_labels)
        kmeans_time = time.time() - start_time

        start_time = time.time()
        optimal_k = find_optimal_kmedoids_clusters(points)
        kmedoids_labels = perform_kmedoids_clustering(points, n_clusters=optimal_k)
        kmedoids_scores = evaluate_clustering(points, kmedoids_labels)
        kmedoids_time = time.time() - start_time

        start_time = time.time()
        optimal_k = find_optimal_agglomerative_clusters(points, max_clusters=10)
        agglomerative_labels = perform_agglomerative_clustering(points, n_clusters=optimal_k)
        agglomerative_scores = evaluate_clustering(points, agglomerative_labels)
        agglomerative_time = time.time() - start_time

        start_time = time.time()
        threshold, branching_factor = find_optimal_birch_params(points)
        birch_labels = perform_birch_clustering(points, threshold=threshold, branching_factor=branching_factor)
        birch_scores = evaluate_clustering(points, birch_labels)
        birch_time = time.time() - start_time

        start_time = time.time()
        eps, min_samples = find_optimal_dbscan_params(points)
        dbscan_labels = perform_dbscan_clustering(points, eps=eps, min_samples=min_samples)
        if len(set(dbscan_labels)) > 1:
            dbscan_scores = evaluate_clustering(points, dbscan_labels)
        else:
            dbscan_scores = {key: None for key in ['SC', 'CH', 'DB', 'CSL', 'DI', 'DB*', 'GD33', 'PB', 'PBM', 'STR']}
        dbscan_time = time.time() - start_time

        start_time = time.time()
        eps, min_samples = find_optimal_optics_params(points)
        optics_labels = perform_optics_clustering(points, eps=eps, min_samples=min_samples)
        if len(set(optics_labels)) > 1:
            optics_scores = evaluate_clustering(points, optics_labels)
        else:
            optics_scores = {key: None for key in ['SC', 'CH', 'DB', 'CSL', 'DI', 'DB*', 'GD33', 'PB', 'PBM', 'STR']}
        optics_time = time.time() - start_time

        start_time = time.time()
        optimal_gmm = find_optimal_gmm_clusters(points)
        gmm_labels = perform_gmm_clustering(points, n_components=optimal_gmm)
        gmm_scores = evaluate_clustering(points, gmm_labels)
        gmm_time = time.time() - start_time
        
        results.update({
            'Patient Type': patient_type,
            'Participant': participant,
            'Stimulus': trial+'-'+stimulus,
            'KMeans Time': kmeans_time,
            'KMeans SC': kmeans_scores['SC'],
            'KMeans CH': kmeans_scores['CH'],
            'KMeans DB': kmeans_scores['DB'],
            'KMeans CSL': kmeans_scores['CSL'],
            'KMeans DI': kmeans_scores['DI'],
            'KMeans DB*': kmeans_scores['DB*'],
            'KMeans GD33': kmeans_scores['GD33'],
            'KMeans PB': kmeans_scores['PB'],
            'KMeans PBM': kmeans_scores['PBM'],
            'KMeans STR': kmeans_scores['STR'],
            'KMedoids Time': kmedoids_time,
            'KMedoids SC': kmedoids_scores['SC'],
            'KMedoids CH': kmedoids_scores['CH'],
            'KMedoids DB': kmedoids_scores['DB'],
            'KMedoids CSL': kmedoids_scores['CSL'],
            'KMedoids DI': kmedoids_scores['DI'],
            'KMedoids DB*': kmedoids_scores['DB*'],
            'KMedoids GD33': kmedoids_scores['GD33'],
            'KMedoids PB': kmedoids_scores['PB'],
            'KMedoids PBM': kmedoids_scores['PBM'],
            'KMedoids STR': kmedoids_scores['STR'],
            'Agglomerative Time': agglomerative_time,
            'Agglomerative SC': agglomerative_scores['SC'],
            'Agglomerative CH': agglomerative_scores['CH'],
            'Agglomerative DB': agglomerative_scores['DB'],
            'Agglomerative CSL': agglomerative_scores['CSL'],
            'Agglomerative DI': agglomerative_scores['DI'],
            'Agglomerative DB*': agglomerative_scores['DB*'],
            'Agglomerative GD33': agglomerative_scores['GD33'],
            'Agglomerative PB': agglomerative_scores['PB'],
            'Agglomerative PBM': agglomerative_scores['PBM'],
            'Agglomerative STR': agglomerative_scores['STR'],
            'BIRCH Time': birch_time,
            'BIRCH SC': birch_scores['SC'],
            'BIRCH CH': birch_scores['CH'],
            'BIRCH DB': birch_scores['DB'],
            'BIRCH CSL': birch_scores['CSL'],
            'BIRCH DI': birch_scores['DI'],
            'BIRCH DB*': birch_scores['DB*'],
            'BIRCH GD33': birch_scores['GD33'],
            'BIRCH PB': birch_scores['PB'],
            'BIRCH PBM': birch_scores['PBM'],
            'BIRCH STR': birch_scores['STR'],
            'DBSCAN Time': dbscan_time,
            'DBSCAN SC': dbscan_scores['SC'],
            'DBSCAN CH': dbscan_scores['CH'],
            'DBSCAN DB': dbscan_scores['DB'],
            'DBSCAN CSL': dbscan_scores['CSL'],
            'DBSCAN DI': dbscan_scores['DI'],
            'DBSCAN DB*': dbscan_scores['DB*'],
            'DBSCAN GD33': dbscan_scores['GD33'],
            'DBSCAN PB': dbscan_scores['PB'],
            'DBSCAN PBM': dbscan_scores['PBM'],
            'DBSCAN STR': dbscan_scores['STR'],
            'OPTICS Time': optics_time,
            'OPTICS SC': optics_scores['SC'],
            'OPTICS CH': optics_scores['CH'],
            'OPTICS DB': optics_scores['DB'],
            'OPTICS CSL': optics_scores['CSL'],
            'OPTICS DI': optics_scores['DI'],
            'OPTICS DB*': optics_scores['DB*'],
            'OPTICS GD33': optics_scores['GD33'],
            'OPTICS PB': optics_scores['PB'],
            'OPTICS PBM': optics_scores['PBM'],
            'OPTICS STR': optics_scores['STR'],
            'GMM Time': gmm_time,
            'GMM SC': gmm_scores['SC'],
            'GMM CH': gmm_scores['CH'],
            'GMM DB': gmm_scores['DB'],
            'GMM CSL': gmm_scores['CSL'],
            'GMM DI': gmm_scores['DI'],
            'GMM DB*': gmm_scores['DB*'],
            'GMM GD33': gmm_scores['GD33'],
            'GMM PB': gmm_scores['PB'],
            'GMM PBM': gmm_scores['PBM'],
            'GMM STR': gmm_scores['STR'],
        })
    
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
