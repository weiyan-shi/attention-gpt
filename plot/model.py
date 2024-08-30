# Given the updated information, let's calculate the parameters based on the provided models and specifications

# Estimated parameter counts
params = {
    'LR': 70,  # Logistic Regression: 70 features
    'SVM': 70,  # SVM: 70 features (linear SVM)
    'KNN': 0,  # KNN: no parameters, only stores training data
    'Decision Tree': 3072,  # Estimation based on the depth and structure, rough estimation
    'Random Forest': 307200,  # 300 trees with an average of ~1024 nodes per tree
    'XGBoost': 307200,  # 300 trees with an average of ~1024 nodes per tree
    'MLP': 17216,  # MLP with 70*128 + 128*64 + 64*1
    'Federated Learning': 10000  # Example estimate for federated learning
}

# Model sizes: Arbitrary scale for demonstration
sizes = {
    'LR': 1,
    'SVM': 2,
    'KNN': 1,
    'Decision Tree': 1.5,
    'Random Forest': 3,
    'XGBoost': 2.5,
    'MLP': 4,
    'Federated Learning': 2
}

# Preparing the data for plotting
model_names = list(params.keys())
param_counts = list(params.values())
model_sizes = list(sizes.values())

# Plotting the data
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3))
plt.scatter(param_counts, model_sizes, color='blue')

# Annotate the points with model names
for i, model in enumerate(model_names):
    plt.text(param_counts[i], model_sizes[i], model, fontsize=9, ha='right')

# Labels and title
plt.xlabel('Number of Parameters')
plt.ylabel('Model Size (Arbitrary Scale)')
plt.title('Model Size vs Number of Parameters')

# Show plot
plt.grid(True)
plt.show()
