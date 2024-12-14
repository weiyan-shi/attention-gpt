import pandas as pd
import matplotlib.pyplot as plt

# Define file paths and corresponding algorithm names
file_paths = [
    "chi25lbw/AOI/analysis/res/cluster.csv",
    "chi25lbw/AOI/analysis/res/heatmap.csv",
    "chi25lbw/AOI/analysis/res/saliency.csv",
    "chi25lbw/AOI/analysis/res/semantic.csv",
]
algorithm_names = ["Cluster", "Heatmap", "Saliency", "Semantic"]

# Initialize a DataFrame to store mean dwell times for all algorithms
combined_mean_dwell_times = pd.DataFrame()

# Read each CSV and extract the mean dwell times for ASD and TD
for algo, file_path in zip(algorithm_names, file_paths):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Group by group (ASD/TD) and calculate mean of mean dwell times for each AOI
    grouped_data = data.groupby("group").mean()

    # Extract only columns related to AOI Mean Dwell Times
    mean_dwell = grouped_data.filter(like="MeanDwellTime")

    # Add algorithm name and reset AOI names to include algorithm-specific identifiers
    mean_dwell = mean_dwell.rename(columns=lambda x: f"{algo}_{x.replace('_MeanDwellTime', '')}")
    mean_dwell["Algorithm"] = algo  # Add algorithm name

    # Append to combined DataFrame
    combined_mean_dwell_times = pd.concat([combined_mean_dwell_times, mean_dwell])

# Reset index to make 'group' a column for plotting
combined_mean_dwell_times.reset_index(inplace=True)

# Melt the DataFrame for easier plotting
melted_data = combined_mean_dwell_times.melt(
    id_vars=["group", "Algorithm"], 
    var_name="AOI", 
    value_name="MeanDwellTime"
)

# Plotting
plt.figure(figsize=(16, 8))

# Use seaborn for grouped bar plots
import seaborn as sns
sns.barplot(
    data=melted_data, 
    x="AOI", 
    y="MeanDwellTime", 
    hue="group", 
    ci="sd", 
    palette="Set2", 
    dodge=True
)

# Add titles and labels
plt.title("Mean Dwell Times Across AOIs for All Algorithms (Different AOI Counts)", fontsize=16)
plt.xlabel("AOIs", fontsize=14)
plt.ylabel("Mean Dwell Time (ms)", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.legend(title="Group", fontsize=12)
plt.tight_layout()

# Save the plot
output_path = "mean_dwell_times_comparison_variable_aoi_counts.png"
plt.savefig(output_path)
plt.show()

print(f"Plot saved as {output_path}")
