import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

# 读取三个CSV文件
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
results_df = results_df.dropna()

# 根据患者类型分组
asd_df = results_df[results_df['Patient Type'] == 'ASD']
td_df = results_df[results_df['Patient Type'] == 'TD']

# 计算各个特征的p值
p_values = {}
for feature in columns_to_keep[1:]:  # Skip 'Patient Type'
    asd_values = asd_df[feature]
    td_values = td_df[feature]
    
    if len(asd_values) > 1 and len(td_values) > 1:
        t_stat, p_val = ttest_ind(asd_values, td_values)
        p_values[feature] = p_val
    else:
        p_values[feature] = np.nan
        print(f"Not enough data for {feature}")

# 将结果转换为数据框并保存
p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'P-Value'])
p_values_df.to_csv('C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\PValues_new.csv', index=False)

print(p_values_df)
