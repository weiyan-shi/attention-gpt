import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

# Define file paths and corresponding algorithm names
file_paths = [
    "chi25lbw/AOI/analysis/res/cluster.csv",
    "chi25lbw/AOI/analysis/res/heatmap.csv",
    "chi25lbw/AOI/analysis/res/saliency.csv",
    "chi25lbw/AOI/analysis/res/semantic.csv",
]
algorithm_names = ["Cluster", "Heatmap", "Saliency", "Semantic"]

# Initialize a DataFrame to store p-values for all features
p_values = pd.DataFrame()

# Loop through each algorithm
for algo, file_path in zip(algorithm_names, file_paths):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Filter data by group
    asd_data = data[data["group"] == "ASD"]
    td_data = data[data["group"] == "TD"]

    # Perform statistical tests for each AOI feature
    for column in data.columns:
        if "MeanDwellTime" in column:
            # Extract feature data for ASD and TD groups
            asd_values = asd_data[column].dropna()
            td_values = td_data[column].dropna()

            # Perform a statistical test (Mann-Whitney U test as an example)
            _, p_value = mannwhitneyu(asd_values, td_values, alternative="two-sided")

            # Append results to the p_values DataFrame
            p_values = p_values.append(
                {"Algorithm": algo, "Feature": column, "p-value": p_value}, ignore_index=True
            )

# Save the p-values to a CSV file
output_path = "feature_p_values.csv"
p_values.to_csv(output_path, index=False)
print(f"P-values saved to {output_path}")
