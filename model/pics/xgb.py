import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb

# Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}
# Accuracy: 0.6554621848739496
#               precision    recall  f1-score   support

#            0       0.62      0.74      0.67        57
#            1       0.71      0.58      0.64        62

#     accuracy                           0.66       119
#    macro avg       0.66      0.66      0.65       119
# weighted avg       0.66      0.66      0.65       119

# AUC: 0.7192982456140351

# 读取CSV文件
file_paths = [
    'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\TrainingDataset\\ClusteringResults_new.csv',
    # 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku-attention\\ClusteringResults_new.csv',
    # 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\woman\\ClusteringResults_new.csv'
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

# 将标签编码为数值
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义参数
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 网格搜索 - XGBoost
grid_xgb = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid_xgb, refit=True, verbose=2, cv=5, scoring='accuracy')
grid_xgb.fit(X_train, y_train)

# 输出最佳参数
print(f"Best Parameters: {grid_xgb.best_params_}")

# 使用最佳参数的模型进行预测
best_model = grid_xgb.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]


# 输出结果
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 计算并输出AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc}")
