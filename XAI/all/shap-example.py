import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import shap

# 读取CSV文件
file = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\ClusteringResults_new.csv'

# 保留指定的列
df = pd.read_csv(file)

# 删除包含缺失值的行
df = df.dropna()

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

X = df[features].values
y = df['Patient Type'].values

# 将标签编码为数值
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 特征归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割数据集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf_model = RandomForestClassifier(
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=300,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 预测和评估
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

# 计算并可视化 SHAP 值
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)

# 总结图
shap.summary_plot(shap_values, X_train, feature_names=features)

# 单个预测解释
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_train[0], feature_names=features)
