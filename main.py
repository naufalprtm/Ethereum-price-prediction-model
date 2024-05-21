import api
import price
import oracle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and preprocess data
data = api.create_dummy_data()
X, y = api.preprocess_data(data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled, X_test_scaled = price.scale_data(X_train, X_test)
X_train_pca = price.dimensionality_reduction(X_train_scaled)
X_test_pca = price.dimensionality_reduction(X_test_scaled)

# Model training and optimization
params = price.bayesian_optimization_search(X_train_pca, y_train)
svr_model = price.train_svr_model(X_train_pca, y_train, params)
nn_model = price.train_neural_network(X_train_pca, y_train)
annealing_params = price.simulated_annealing_search(X_train_pca, y_train, X_test_pca, y_test)
annealing_svr_model = price.train_svr_model(X_train_pca, y_train, {'C': annealing_params[0], 'epsilon': annealing_params[1]})

# Select the best model
models = [svr_model, nn_model, annealing_svr_model]
best_model, best_metrics = oracle.select_best_model(models, X_test_pca, y_test)

# Monte Carlo Simulation
mc_simulations = price.monte_carlo_simulation(best_model, X_test_pca)
mean_simulated_price = mc_simulations.mean(axis=0)

# Exploring Patterns
gmm, silhouette_avg, db_index, rand_index, ch_index, completeness, homogeneity = oracle.explore_patterns(X_train_pca, y_train)

# Output results
print("=== Results ===")
print(f"Best Model: {best_model}")
print("Best Model Metrics:")
for metric, value in best_metrics.items():
    print(f"- {metric}: {value}")

print("\n=== Clustering Analysis ===")
print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Index: {db_index}")
if rand_index is not None:
    print(f"Adjusted Rand Index: {rand_index}")
print(f"Calinski-Harabasz Index: {ch_index}")
if completeness is not None:
    print(f"Completeness Score: {completeness}")
if homogeneity is not None:
    print(f"Homogeneity Score: {homogeneity}")

# Plotting
plt.figure(figsize=(14, 6))

# Plot 1: Actual vs Simulated Prices
plt.subplot(1, 2, 1)
plt.plot(data['Date'], data['Price'], label='Actual Prices', color='blue')
plt.plot(data['Date'].iloc[X_test_pca[:, 0].argsort()], mean_simulated_price[X_test_pca[:, 0].argsort()], label='Simulated Prices', color='red')
plt.title("Actual vs Simulated Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

# Plot 2: Clustering of Training Data
plt.subplot(1, 2, 2)
if len(X_train_pca[0]) == 1:
    plt.scatter(X_train_pca, [0] * len(X_train_pca), c=y_train, cmap='viridis')
    plt.title("Clustering of Training Data")
    plt.xlabel("Principal Component 1")
else:
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
    plt.title("Clustering of Training Data (First 2 Principal Components)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
plt.colorbar(label='Price')

# Annotate Clusters
for i, txt in enumerate(y_train):
    if len(X_train_pca[0]) == 1:
        plt.annotate(txt, (X_train_pca[i], 0), textcoords="offset points", xytext=(0,10), ha='center')
    else:
        plt.annotate(txt, (X_train_pca[i, 0], X_train_pca[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()
