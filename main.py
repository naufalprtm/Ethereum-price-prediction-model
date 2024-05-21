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
best_model = oracle.select_best_model(models, X_test_pca, y_test)

# Monte Carlo Simulation
mc_simulations = price.monte_carlo_simulation(best_model, X_test_pca)
mean_simulated_price = mc_simulations.mean(axis=0)

# Exploring Patterns
gmm = oracle.explore_patterns(X_train_pca)

# Output results
print(f"Best Model: {best_model}")
print(f"Mean Simulated Prices: {mean_simulated_price}")

plt.plot(data['Date'], data['Price'], label='Actual Prices')
plt.plot(data['Date'].iloc[X_test_pca[:, 0].argsort()], mean_simulated_price[X_test_pca[:, 0].argsort()], label='Simulated Prices')
plt.legend()
plt.show()
