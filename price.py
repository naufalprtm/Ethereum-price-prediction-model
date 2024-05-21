from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import dual_annealing
from bayes_opt import BayesianOptimization
import numpy as np

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def dimensionality_reduction(X):
    pca = PCA(n_components=1)
    return pca.fit_transform(X)

def bayesian_optimization_search(X_train, y_train):
    def svr_cv(C, epsilon):
        val = cross_val_score(
            SVR(C=C, epsilon=epsilon),
            X_train, y_train,
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

def train_svr_model(X_train, y_train, params):
    svr = SVR(C=params['C'], epsilon=params['epsilon'])
    svr.fit(X_train, y_train)
    return svr

def train_neural_network(X_train, y_train):
    nn = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    nn.fit(X_train, y_train)
    return nn

def simulated_annealing_search(X_train, y_train, X_test, y_test):
    def objective(params):
        C, epsilon = params
        model = SVR(C=C, epsilon=epsilon)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return mean_squared_error(y_test, pred)

    bounds = [(0.1, 10), (0.01, 1)]
    result = dual_annealing(objective, bounds)
    return result.x

def monte_carlo_simulation(model, X_test, num_simulations=100):
    predictions = []
    for _ in range(num_simulations):
        simulated_prices =
