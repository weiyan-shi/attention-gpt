import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

# 读取CSV文件
results_df = pd.read_csv('C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku-attention\\ClusteringResults.csv')

# 删除包含缺失值的行
results_df = results_df.dropna()

# 根据患者类型分组
asd_df = results_df[results_df['Patient Type'] == 'asd']
td_df = results_df[results_df['Patient Type'] == 'td']

# 要计算的特征列表
features = [
    'KMeans Silhouette Score', 'KMeans CH Score', 'KMeans DB Score',
    'DBSCAN Silhouette Score', 'DBSCAN CH Score', 'DBSCAN DB Score', 'DBSCAN Noise Ratio',
    'GMM Silhouette Score', 'GMM CH Score', 'GMM DB Score',
    'BIRCH Silhouette Score', 'BIRCH CH Score', 'BIRCH DB Score'
]

# 计算各个特征的p值
p_values = {}
for feature in features:
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
p_values_df.to_csv('C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku-attention\\PValues.csv', index=False)

print(p_values_df)
