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
combined_total_duration = pd.DataFrame()

# Read each CSV and extract the mean dwell times for ASD and TD
for algo, file_path in zip(algorithm_names, file_paths):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Group by group (ASD/TD) and calculate mean of mean dwell times for each AOI
    grouped_data = data.groupby("group").mean()

    # Extract only columns related to AOI Mean Dwell Times
    total_duration = grouped_data.filter(like="TotalDuration")

    # Add algorithm name and reset AOI names to include algorithm-specific identifiers
    total_duration = total_duration.rename(columns=lambda x: f"{algo}_{x.replace('_TotalDuration', '')}")
    total_duration["Algorithm"] = algo  # Add algorithm name

    # Append to combined DataFrame
    combined_total_duration = pd.concat([combined_total_duration, total_duration])

# Reset index to make 'group' a column for plotting
combined_total_duration.reset_index(inplace=True)

# Melt the DataFrame for easier plotting
melted_data = combined_total_duration.melt(
    id_vars=["group", "Algorithm"], 
    var_name="AOI", 
    value_name="TotalDuration"
)

# Plotting
plt.figure(figsize=(16, 8))

# Use seaborn for grouped bar plots
import seaborn as sns
sns.barplot(
    data=melted_data, 
    x="AOI", 
    y="TotalDuration", 
    hue="group", 
    ci="sd", 
    palette="Set2", 
    dodge=True
)

# Add titles and labels
plt.title("Total Duration Across AOIs for All Algorithms (Different AOI Counts)", fontsize=16)
plt.xlabel("AOIs", fontsize=14)
plt.ylabel("Total Duration", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.legend(title="Group", fontsize=12)
plt.tight_layout()

# Save the plot
output_path = "total_duration_comparison_variable_aoi_counts.png"
plt.savefig(output_path)
plt.show()

print(f"Plot saved as {output_path}")
