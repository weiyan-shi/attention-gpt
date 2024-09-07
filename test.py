import pandas as pd
# Creating a dictionary with the data provided
data = {
    'Algorithm': ['Random Forest', 'XGBoost', 'MLP', 'Random Forest', 'XGBoost', 'MLP', 'Random Forest', 'XGBoost', 'MLP'],
    'Dataset': ['Qiao', 'Qiao', 'Qiao', 'Cillia', 'Cillia', 'Cillia', 'Saliency4ASD', 'Saliency4ASD', 'Saliency4ASD'],
    'Accuracy': [0.725, 0.732, 0.744, 0.733, 0.735, 0.701, 0.745, 0.712, 0.681],
    'Precision': [0.715, 0.734, 0.743, 0.742, 0.741, 0.682, 0.744, 0.711, 0.684],
    'Recall': [0.726, 0.731, 0.742, 0.735, 0.732, 0.691, 0.746, 0.712, 0.683],
    'F1-score': [0.711, 0.732, 0.741, 0.701, 0.714, 0.671, 0.744, 0.714, 0.682],
    'AUC': [0.791, 0.802, 0.741, 0.813, 0.784, 0.671, 0.834, 0.783, 0.682]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Calculate the average for each algorithm
avg_df = df.groupby('Algorithm').mean()

print(avg_df)

# # Display the averages to the user
# import ace_tools as tools; tools.display_dataframe_to_user(name="Average Performance Across Datasets", dataframe=avg_df)
