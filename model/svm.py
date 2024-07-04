# Best Parameters: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
# Model: SVC with best parameters
# Accuracy: 0.6495190296946884
#               precision    recall  f1-score   support

#          ASD       0.66      0.79      0.72      1346
#           TD       0.63      0.47      0.54      1045

#     accuracy                           0.65      2391
#    macro avg       0.65      0.63      0.63      2391
# weighted avg       0.65      0.65      0.64      2391


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 读取CSV文件
file_paths = [
    'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\TrainingDataset\\ClusteringResults_new.csv',
    'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku-attention\\ClusteringResults_new.csv',
    'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\woman\\ClusteringResults_new.csv'
]

# 要保留的列
columns_to_keep = [
    'Patient Type',
    'KMeans Silhouette Score', 'KMeans CH Score', 'KMeans DB Score',
    'DBSCAN Silhouette Score', 'DBSCAN CH Score', 'DBSCAN DB Score',
    'GMM Silhouette Score', 'GMM CH Score', 'GMM DB Score',
    'BIRCH Silhouette Score', 'BIRCH CH Score', 'BIRCH DB Score',
    'Agglomerative Silhouette Score', 'Agglomerative CH Score', 'Agglomerative DB Score',
    'KMedoids Silhouette Score', 'KMedoids CH Score', 'KMedoids DB Score',
    'OPTICS Silhouette Score', 'OPTICS CH Score', 'OPTICS DB Score'
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
    'KMeans Silhouette Score', 'KMeans CH Score', 'KMeans DB Score',
    'DBSCAN Silhouette Score', 'DBSCAN CH Score', 'DBSCAN DB Score',
    'GMM Silhouette Score', 'GMM CH Score', 'GMM DB Score',
    'BIRCH Silhouette Score', 'BIRCH CH Score', 'BIRCH DB Score',
    'Agglomerative Silhouette Score', 'Agglomerative CH Score', 'Agglomerative DB Score',
    'KMedoids Silhouette Score', 'KMedoids CH Score', 'KMedoids DB Score',
    'OPTICS Silhouette Score', 'OPTICS CH Score', 'OPTICS DB Score'
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
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# 网格搜索
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5, n_jobs=-1, scoring='accuracy')
grid.fit(X_train, y_train)

# 输出最佳参数
print(f"Best Parameters: {grid.best_params_}")

# 使用最佳参数的模型进行预测
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# 输出结果
print(f"Model: SVC with best parameters")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
