import scipy.io as sio
import pandas as pd
import numpy as np


# Load the MATLAB file
mat_file = 'C:\\Users\\86178\\Desktop\\attention-gpt\\dataset\\pku-attention\\HFA_Group_2.mat'
mat = sio.loadmat(mat_file)
eyeMovementsASD = mat['eyeMovementsASD']

mat_file = 'C:\\Users\\86178\\Desktop\\attention-gpt\\movie\\RawEyetrackingTD.mat'
mat = sio.loadmat(mat_file)
eyeMovementsTD = mat['eyeMovements']


# 初始化一个空的列表用于存储数据
data = []

# 将ASD数据转换为二维
asd_participant_count, movie_count, direction_count, time_count = eyeMovementsASD.shape

for participant in range(asd_participant_count):
    for movie in range(movie_count):
        for time in range(time_count):
            x_value = eyeMovementsASD[participant, movie, 0, time]
            y_value = eyeMovementsASD[participant, movie, 1, time]
            data.append([
                participant + 1,
                movie + 1,
                time + 1,
                x_value,
                y_value,
                'ASD'
            ])

# 将TD数据转换为二维
td_participant_count = eyeMovementsTD.shape[0]

for participant in range(td_participant_count):
    for movie in range(movie_count):
        for time in range(time_count):
            x_value = eyeMovementsTD[participant, movie, 0, time]
            y_value = eyeMovementsTD[participant, movie, 1, time]
            data.append([
                asd_participant_count + participant + 1,
                movie + 1,
                time + 1,
                x_value,
                y_value,
                'TD'
            ])

# 将列表转换为DataFrame
columns = ['Participant', 'Movie', 'Time', 'x', 'y', 'Type']
df = pd.DataFrame(data, columns=columns)

# 将DataFrame保存为CSV文件
df.to_csv('eye_movements.csv', index=False)

print("数据已成功转换为CSV格式并保存为 eye_movements.csv")