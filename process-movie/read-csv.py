import scipy.io as sio
import pandas as pd
import numpy as np

# # Load the MATLAB file
# mat_file = 'C:\\Users\\86178\\Desktop\\attention-gpt\\pku-attention\\LFA_Group.mat'
# data = sio.loadmat(mat_file)

# # Extract the numpy array
# sub_array = data['sub']

# # Check the structure of the array
# print(f"sub_array type: {type(sub_array)}")
# print(f"sub_array shape: {sub_array.shape}")

# # Convert to a DataFrame ensuring all data is in a suitable format
# # We use np.vectorize to ensure each element is properly converted to a string if necessary
# sub_array = np.vectorize(lambda x: str(x) if not isinstance(x, (int, float)) else x)(sub_array)

# # Convert the numpy array to a pandas DataFrame
# df = pd.DataFrame(sub_array)

# # Save the DataFrame to a CSV file
# csv_file = 'C:\\Users\\86178\\Desktop\\attention-gpt\\pku-attention\\LFA_Group.csv'
# df.to_csv(csv_file, index=False)

# print(f"Data saved to {csv_file}")

# Load the MATLAB file
mat_file = 'C:\\Users\\86178\\Desktop\\attention-gpt\\movie\\RawEyetrackingASD.mat'
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