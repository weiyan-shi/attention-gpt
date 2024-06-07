import os
import json
import pandas as pd

# 读取csv文件
csv_file_path = '../dataset.csv'
df = pd.read_csv(csv_file_path)

# 遍历文件夹
folder_path = '../DREAMdataset'
for root, dirs, files in os.walk(folder_path):
    # 检查是否是最底层目录
    if not dirs:
        for file in files:
            # 检查文件名是否在csv中
            if file.endswith('.json') and file in df['filename'].values:
                json_file_path = os.path.join(root, file)
                
                # 读取json文件
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                
                # 提取需要的字段
                condition = data.get('condition', None)
                ability = None
                difficultyLevel = None
                
                task = data.get('task', {})
                if isinstance(task, list):
                    if len(task) > 0:
                        ability = task[0].get('ability', None)
                        difficultyLevel = task[0].get('difficultyLevel', None)
                else:
                    ability = task.get('ability', None)
                    difficultyLevel = task.get('difficultyLevel', None)
                
                # 更新csv文件中的数据
                df.loc[df['filename'] == file, 'condition'] = condition
                df.loc[df['filename'] == file, 'ability'] = ability
                df.loc[df['filename'] == file, 'difficultyLevel'] = difficultyLevel

# 保存更新后的csv文件
df.to_csv('../dataset-v1.csv', index=False)
