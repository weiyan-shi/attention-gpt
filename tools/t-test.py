import os
import pandas as pd
from scipy import stats


# 设定包含CSV文件的文件夹路径
folder_path = 'score'

# 初始化一个空的DataFrame列表来存放每个文件的数据
dfs = []

# 遍历文件夹内的每一个文件
for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为.csv
    if filename.endswith('.csv'):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 读取CSV文件，并添加到列表中
        dfs.append(pd.read_csv(file_path))

# 使用concat函数合并所有的DataFrame对象
df = pd.concat(dfs, ignore_index=True)

df_ja = df[df['task'] == 'JA']

# 进一步分割出包含 'Initial diagnosis' 和 'Intervention' 的两组数据
df_initial = df_ja[df_ja['filename'].str.contains('Initial diagnosis')]
df_intervention = df_ja[df_ja['filename'].str.contains('Intervention')]

# 对于 'd_db', 'd_s', 'd_ch', 'd_n_pc' 执行 t-test 并获取 p 值
columns_to_test = ['d_db', 'd_s', 'd_ch', 'd_n_pc']
p_values = {}

for col in columns_to_test:
    # 确保数据没有缺失值，否则 t-test 无法运行
    clean_initial = df_initial[col].dropna()
    clean_intervention = df_intervention[col].dropna()
    
    # 执行 t-test
    t_stat, p_val = stats.ttest_ind(clean_initial, clean_intervention)
    
    # 将结果存储到字典里
    p_values[col] = p_val

# 输出 p 值结果
print(p_values)

df_im = df[df['task'] == 'IM']

# 进一步分割出包含 'Initial diagnosis' 和 'Intervention' 的两组数据
df_initial = df_im[df_im['filename'].str.contains('Initial diagnosis')]
df_intervention = df_im[df_im['filename'].str.contains('Intervention')]

# 对于 'd_db', 'd_s', 'd_ch', 'd_n_pc' 执行 t-test 并获取 p 值
columns_to_test = ['d_db', 'd_s', 'd_ch', 'd_n_pc']
p_values = {}

for col in columns_to_test:
    # 确保数据没有缺失值，否则 t-test 无法运行
    clean_initial = df_initial[col].dropna()
    clean_intervention = df_intervention[col].dropna()
    
    # 执行 t-test
    t_stat, p_val = stats.ttest_ind(clean_initial, clean_intervention)
    
    # 将结果存储到字典里
    p_values[col] = p_val

# 输出 p 值结果
print(p_values)

df_tt = df[df['task'] == 'TT']

# 进一步分割出包含 'Initial diagnosis' 和 'Intervention' 的两组数据
df_initial = df_tt[df_tt['filename'].str.contains('Initial diagnosis')]
df_intervention = df_tt[df_tt['filename'].str.contains('Intervention')]

# 对于 'd_db', 'd_s', 'd_ch', 'd_n_pc' 执行 t-test 并获取 p 值
columns_to_test = ['d_db', 'd_s', 'd_ch', 'd_n_pc']
p_values = {}

for col in columns_to_test:
    # 确保数据没有缺失值，否则 t-test 无法运行
    clean_initial = df_initial[col].dropna()
    clean_intervention = df_intervention[col].dropna()
    
    # 执行 t-test
    t_stat, p_val = stats.ttest_ind(clean_initial, clean_intervention)
    
    # 将结果存储到字典里
    p_values[col] = p_val

# 输出 p 值结果
print(p_values)
