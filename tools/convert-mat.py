import scipy.io as sio
import pandas as pd
import numpy as np

# Load the MATLAB file
mat_file = 'C:\\Users\\86178\\Desktop\\attention-gpt\\pku-attention\\LFA_Group.mat'
data = sio.loadmat(mat_file)

# Extract the numpy array
sub_array = data['sub']

# Check the structure of the array
print(f"sub_array type: {type(sub_array)}")
print(f"sub_array shape: {sub_array.shape}")

# Convert to a DataFrame ensuring all data is in a suitable format
# We use np.vectorize to ensure each element is properly converted to a string if necessary
sub_array = np.vectorize(lambda x: str(x) if not isinstance(x, (int, float)) else x)(sub_array)

# Convert the numpy array to a pandas DataFrame
df = pd.DataFrame(sub_array)

# Save the DataFrame to a CSV file
csv_file = 'C:\\Users\\86178\\Desktop\\attention-gpt\\pku-attention\\LFA_Group.csv'
df.to_csv(csv_file, index=False)

print(f"Data saved to {csv_file}")



# for key in data.keys():
#     if not key.startswith('__'):
#         print(f"Key: {key}, Type: {type(data[key])}, Shape: {data[key].shape}")
#         print(data[key])
 
# # 提取数据
# data_array = data['eyeMovementsASD']
 
# # 转换为DataFrame
# df = pd.DataFrame(data_array)
 
# # 保存为Excel文件
# csv_file = 'eyeMovementsASD.csv'
# df.to_csv(csv_file, index=False)