import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 读取三个CSV文件
file_paths = [
    'C:\\Users\\86178\\Desktop\\attention-gpt\\TrainingDataset\\ClusteringResults.csv',
    'C:\\Users\\86178\\Desktop\\attention-gpt\\movie\\ClusteringResults.csv',
    'C:\\Users\\86178\\Desktop\\attention-gpt\\woman\\ClusteringResults.csv'
]

# 要保留的列
columns_to_keep = [
    'Patient Type', 'KMeans Silhouette Score', 'KMeans CH Score', 'KMeans DB Score',
    'DBSCAN Silhouette Score', 'DBSCAN CH Score', 'DBSCAN DB Score', 'DBSCAN Noise Ratio',
    'GMM Silhouette Score', 'GMM CH Score', 'GMM DB Score',
    'BIRCH Silhouette Score', 'BIRCH CH Score', 'BIRCH DB Score'
]

# 合并CSV文件
dfs = [pd.read_csv(file) for file in file_paths]
results_df = pd.concat(dfs, join='outer', ignore_index=True)

# 保留指定的列
results_df = results_df[columns_to_keep]

# 删除包含缺失值的行
data = results_df.dropna()

target = 'Patient Type'


# 映射 'ASD' 为 1，'TD' 为 0
data['Patient Type'] = data['Patient Type'].map({'ASD': 1, 'TD': 0})

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 使用 Autogluon 进行自动学习
label = 'Patient Type'
time_limit = 1800
num_trials = 5
search_strategy = 'auto'

# 定义超参数
# Extra Trees with Gini impurity criterion
extratrees_options = {
    'n_estimators': 10,
    'criterion': 'gini',
    'max_depth': None
}

randomforest_options = {
    'n_estimators': 10,     # Fewer trees can help to prevent overfitting
    'max_depth': 3,         # A shallow depth to prevent the model from learning the data too closely
}

catboost_options = {
    'iterations': 10,
    'depth': 3,
    'learning_rate': 0.1
}

lightgbm_options = {
    'num_boost_round': 10,
    'num_leaves': 8,
    'learning_rate': 0.1
}

# Then you would pass these to AutoGluon like so:
hyperparameters = {
    'XT': extratrees_options,
    'RF': randomforest_options,
    'CAT': catboost_options,
    'GBM': lightgbm_options
}

# 设置超参数调整相关参数
hyperparameter_tune_kwargs = {'num_trials': num_trials, 'scheduler': 'local', 'searcher': search_strategy}

# Fit the model with hyperparameter tuning
predictor = TabularPredictor(label=label, path='self_designed', eval_metric='roc_auc').fit(train_data, time_limit=time_limit, num_stack_levels=1, num_bag_folds=3,
                                              hyperparameters=hyperparameters,
                                              hyperparameter_tune_kwargs=hyperparameter_tune_kwargs)

# predictor.save('saved_model_2')


# Make predictions on the test set
y_pred = predictor.predict(test_data.drop(columns=[target]))

# Make predictions on the test set (using predict_proba)
y_proba = predictor.predict_proba(test_data.drop(columns=[target]))

# Extract probabilities for the positive class (class 1)
y_pred_proba = y_proba.iloc[:, 1]

# Evaluate model performance using ROC AUC
roc_auc = roc_auc_score(test_data[target], y_pred_proba)

# Evaluate model performance
accuracy = accuracy_score(test_data[target], y_pred)
precision = precision_score(test_data[target], y_pred)
recall = recall_score(test_data[target], y_pred)
f1 = f1_score(test_data[target], y_pred)

print(f'Test Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC Score: {roc_auc}')



