import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, kstest, norm
import numpy as np

# 读取CSV文件
results_df = pd.read_csv('C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pics\\ClusteringResults_new.csv')

# 删除包含缺失值的行
results_df = results_df.dropna()

# 根据患者类型分组
asd_df = results_df[results_df['Patient Type'] == 'ASD']
td_df = results_df[results_df['Patient Type'] == 'TD']

# 要计算的特征列表
features = [
    'KMeans Silhouette Score', 'KMeans CH Score', 'KMeans DB Score',
    'DBSCAN Silhouette Score', 'DBSCAN CH Score', 'DBSCAN DB Score',
    'GMM Silhouette Score', 'GMM CH Score', 'GMM DB Score',
    'BIRCH Silhouette Score', 'BIRCH CH Score', 'BIRCH DB Score',
    'Agglomerative Silhouette Score', 'Agglomerative CH Score', 'Agglomerative DB Score',
    'KMedoids Silhouette Score', 'KMedoids CH Score', 'KMedoids DB Score',
    'OPTICS Silhouette Score', 'OPTICS CH Score', 'OPTICS DB Score'
]

# 计算各个特征的p值
p_values = {}
for feature in features:  # Skip 'Patient Type'
    asd_values = asd_df[feature]
    td_values = td_df[feature]
    
    if len(asd_values) > 1 and len(td_values) > 1:
        # 正态性检验
        asd_normal = kstest(asd_values, 'norm', args=(asd_values.mean(), asd_values.std()))[1] > 0.05
        td_normal = kstest(td_values, 'norm', args=(td_values.mean(), td_values.std()))[1] > 0.05
        
        # if asd_normal and td_normal:
        #     # 如果数据服从正态分布，使用t-test
        #     t_stat, p_val = ttest_ind(asd_values, td_values)
        #     print('符合')
        # else:
            # 如果数据不服从正态分布，使用曼-惠特尼U检验
        u_stat, p_val = mannwhitneyu(asd_values, td_values)
            # print('不符合')
        
        p_values[feature] = p_val
    else:
        p_values[feature] = np.nan
        print(f"Not enough data for {feature}")

# 将结果转换为数据框并保存
p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'P-Value'])

# 应用Holm-Bonferroni校正
p_values_df = p_values_df.sort_values(by='P-Value')
m = len(p_values_df)
corrected_p_values = []
for i, (feature, p) in enumerate(p_values_df.values):
    corrected_p = p * (m - i)
    corrected_p_values.append(min(corrected_p, 1.0))  # Holm-Bonferroni校正

p_values_df['Corrected P-Value'] = corrected_p_values

p_values_df.to_csv('C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pics\\PValues_drq.csv', index=False)

print(p_values_df)
