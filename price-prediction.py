import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.mixture import GaussianMixture
import random

# Load your data
# data = pd.read_csv('eth_data.csv')

# For the purpose of this example, let's create dummy data
def create_dummy_data():
    dates = pd.date_range(start='1/1/2020', periods=100)
    prices = np.random.rand(100) * 1000
    return pd.DataFrame({'Date': dates, 'Price': prices})

data = create_dummy_data()

# Preprocessing
X = data[['Date']].apply(lambda x: x.factorize()[0]).values.reshape(-1, 1)  # Converting date to categorical integer
y = data['Price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dimensionality Reduction
def dimensionality_reduction(X):
    pca = PCA(n_components=1)
    return pca.fit_transform(X)

X_train_pca = dimensionality_reduction(X_train)
X_test_pca = dimensionality_reduction(X_test)

# Bayesian Optimization Search for Hyperparameter Tuning
def bayesian_optimization_search():
    def svr_cv(C, epsilon):
        val = cross_val_score(
            SVR(C=C, epsilon=epsilon),
            X_train_pca, y_train,
            scoring='neg_mean_squared_error',
            cv=5
        ).mean()
        return val

    optimizer = BayesianOptimization(
        f=svr_cv,
        pbounds={"C": (0.1, 10), "epsilon": (0.01, 1)},
        random_state=42,
    )
    optimizer.maximize(n_iter=10)
    return optimizer.max['params']

# Train SVR Model
def train_svr_model(params):
    svr = SVR(C=params['C'], epsilon=params['epsilon'])
    svr.fit(X_train_pca, y_train)
    return svr

# Train Neural Network
def train_neural_network():
    nn = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    nn.fit(X_train_pca, y_train)
    return nn

# Simulated Annealing Search
def simulated_annealing_search():
    def objective(params):
        C, epsilon = params
        model = SVR(C=C, epsilon=epsilon)
        model.fit(X_train_pca, y_train)
        pred = model.predict(X_test_pca)
        return mean_squared_error(y_test, pred)

    bounds = [(0.1, 10), (0.01, 1)]
    result = dual_annealing(objective, bounds)
    return result.x

# Monte Carlo Simulation for Price Estimation
def monte_carlo_simulation(model, num_simulations=100):
    predictions = []
    for _ in range(num_simulations):
        simulated_prices = model.predict(X_test_pca) + np.random.normal(0, 0.1, len(y_test))
        predictions.append(simulated_prices)
    return np.array(predictions)

# Explore Patterns using GMM
def explore_patterns(X):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    return gmm

# Select Best Model based on Performance
def select_best_model(models):
    best_model = None
    best_score = float('inf')
    for model in models:
        pred = model.predict(X_test_pca)
        score = mean_squared_error(y_test, pred)
        if score < best_score:
            best_score = score
            best_model = model
    return best_model

# Main Execution
params = bayesian_optimization_search()
svr_model = train_svr_model(params)
nn_model = train_neural_network()
annealing_params = simulated_annealing_search()
annealing_svr_model = train_svr_model({'C': annealing_params[0], 'epsilon': annealing_params[1]})

models = [svr_model, nn_model, annealing_svr_model]
best_model = select_best_model(models)

# Monte Carlo Simulation
mc_simulations = monte_carlo_simulation(best_model)
mean_simulated_price = mc_simulations.mean(axis=0)

# Exploring Patterns
gmm = explore_patterns(X_train_pca)

# Output results
print(f"Best Model: {best_model}")
print(f"Mean Simulated Prices: {mean_simulated_price}")

plt.plot(data['Date'], data['Price'], label='Actual Prices')
plt.plot(data['Date'].iloc[X_test_pca[:, 0].argsort()], mean_simulated_price[X_test_pca[:, 0].argsort()], label='Simulated Prices')
plt.legend()
plt.show()
