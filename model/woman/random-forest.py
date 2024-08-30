from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Best Parameters: {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
# Accuracy: 0.7195357833655706
#               precision    recall  f1-score   support

#          ASD       0.79      0.38      0.52       201
#           TD       0.70      0.93      0.80       316

#     accuracy                           0.72       517
#    macro avg       0.74      0.66      0.66       517
# weighted avg       0.74      0.72      0.69       517

# AUC: 0.7686724604823981

# 读取CSV文件
file_paths = [
    # 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\TrainingDataset\\ClusteringResults_new.csv',
    # 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku-attention\\ClusteringResults_new.csv',
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
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 3, 5, 7, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 网格搜索
grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=2, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# 输出最佳参数
print(f"Best Parameters: {grid.best_params_}")

# 使用最佳参数的模型进行预测
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]


# 输出结果
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 计算并输出AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc}")
