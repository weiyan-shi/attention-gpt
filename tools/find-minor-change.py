import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 数据载入
df = pd.read_csv('../score/User 3.csv')

# 查看数据基本情况
print(df.head())

# 分组统计，例如计算每个组的平均值
grouped = df.groupby(['condition', 'ability', 'difficulty_level'])['d_db', 'd_s', 'd_ch', 'd_n_pc'].mean()

# 输出结果查看
print(grouped)


# 可视化示例，画出某些变量的分布
sns.boxplot(x='condition', y='d_db', hue='difficulty_level', data=df)
plt.show()
