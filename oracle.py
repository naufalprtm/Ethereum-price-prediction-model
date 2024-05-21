from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
import numpy as np

def explore_patterns(X):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    return gmm

def select_best_model(models, X_test, y_test):
    best_model = None
    best_score = float('inf')
    for model in models:
        pred = model.predict(X_test)
        score = mean_squared_error(y_test, pred)
        if score < best_score:
            best_score = score
            best_model = model
    return best_model
