from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import time

# Best Parameters: {'C': 0.1, 'max_iter': 200, 'solver': 'lbfgs'}
# Accuracy: 0.719088319088319
#               precision    recall  f1-score   support

#          asd       0.75      0.83      0.79      1090
#           td       0.66      0.54      0.59       665

#     accuracy                           0.72      1755
#    macro avg       0.70      0.68      0.69      1755
# weighted avg       0.71      0.72      0.71      1755

# AUC: 0.7768324480927088
# Prediction execution time: 0.0009980201721191406 seconds

# 读取CSV文件
file_paths = [
    'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku\\ClusteringResults_10.csv',
    # 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku-attention\\ClusteringResults_new.csv',
    # 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\woman\\ClusteringResults_new.csv'
]

# 要保留的列
columns_to_keep = [
    'Patient Type',
    'KMeans SC', 'KMeans CH', 'KMeans DB', 'KMeans CSL', 'KMeans DI', 'KMeans DB*', 'KMeans GD33', 'KMeans PB', 'KMeans PBM', 'KMeans STR',
    'KMedoids SC', 'KMedoids CH', 'KMedoids DB', 'KMedoids CSL', 'KMedoids DI', 'KMedoids DB*', 'KMedoids GD33', 'KMedoids PB', 'KMedoids PBM', 'KMedoids STR',
    'Agglomerative SC', 'Agglomerative CH', 'Agglomerative DB', 'Agglomerative CSL', 'Agglomerative DI', 'Agglomerative DB*', 'Agglomerative GD33', 'Agglomerative PB', 'Agglomerative PBM', 'Agglomerative STR',
    'BIRCH SC', 'BIRCH CH', 'BIRCH DB', 'BIRCH CSL', 'BIRCH DI', 'BIRCH DB*', 'BIRCH GD33', 'BIRCH PB', 'BIRCH PBM', 'BIRCH STR',
    'DBSCAN SC', 'DBSCAN CH', 'DBSCAN DB', 'DBSCAN CSL', 'DBSCAN DI', 'DBSCAN DB*', 'DBSCAN GD33', 'DBSCAN PB', 'DBSCAN PBM', 'DBSCAN STR',
    'OPTICS SC', 'OPTICS CH', 'OPTICS DB', 'OPTICS CSL', 'OPTICS DI', 'OPTICS DB*', 'OPTICS GD33', 'OPTICS PB', 'OPTICS PBM', 'OPTICS STR',
    'GMM SC', 'GMM CH', 'GMM DB', 'GMM CSL', 'GMM DI', 'GMM DB*', 'GMM GD33', 'GMM PB', 'GMM PBM', 'GMM STR'
]

# 合并CSV文件
dfs = [pd.read_csv(file) for file in file_paths]
results_df = pd.concat(dfs, join='outer', ignore_index=True)

# 保留指定的列
results_df = results_df[columns_to_keep]

# 删除包含缺失值的行
df = results_df.dropna()

# 提取特征和标签
features = [
    'KMeans SC', 'KMeans CH', 'KMeans DB', 'KMeans CSL', 'KMeans DI', 'KMeans DB*', 'KMeans GD33', 'KMeans PB', 'KMeans PBM', 'KMeans STR',
    'KMedoids SC', 'KMedoids CH', 'KMedoids DB', 'KMedoids CSL', 'KMedoids DI', 'KMedoids DB*', 'KMedoids GD33', 'KMedoids PB', 'KMedoids PBM', 'KMedoids STR',
    'Agglomerative SC', 'Agglomerative CH', 'Agglomerative DB', 'Agglomerative CSL', 'Agglomerative DI', 'Agglomerative DB*', 'Agglomerative GD33', 'Agglomerative PB', 'Agglomerative PBM', 'Agglomerative STR',
    'BIRCH SC', 'BIRCH CH', 'BIRCH DB', 'BIRCH CSL', 'BIRCH DI', 'BIRCH DB*', 'BIRCH GD33', 'BIRCH PB', 'BIRCH PBM', 'BIRCH STR',
    'DBSCAN SC', 'DBSCAN CH', 'DBSCAN DB', 'DBSCAN CSL', 'DBSCAN DI', 'DBSCAN DB*', 'DBSCAN GD33', 'DBSCAN PB', 'DBSCAN PBM', 'DBSCAN STR',
    'OPTICS SC', 'OPTICS CH', 'OPTICS DB', 'OPTICS CSL', 'OPTICS DI', 'OPTICS DB*', 'OPTICS GD33', 'OPTICS PB', 'OPTICS PBM', 'OPTICS STR',
    'GMM SC', 'GMM CH', 'GMM DB', 'GMM CSL', 'GMM DI', 'GMM DB*', 'GMM GD33', 'GMM PB', 'GMM PBM', 'GMM STR'
]

X = df[features]
y = df['Patient Type']

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义参数网格
param_grid = {
    'C': [0.1],
    'solver': ['lbfgs'],
    'max_iter': [200]
}

# 网格搜索
grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=2, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# 输出最佳参数
print(f"Best Parameters: {grid.best_params_}")

# 使用最佳参数的模型进行预测
best_model = grid.best_estimator_
start_time = time.time()
y_pred = best_model.predict(X_test)
end_time = time.time()
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 输出结果
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 计算并输出AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc}")

execution_time = end_time - start_time
print(f"Prediction execution time: {execution_time} seconds")
