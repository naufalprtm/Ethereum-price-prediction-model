from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, silhouette_score, davies_bouldin_score, adjusted_rand_score, calinski_harabasz_score, completeness_score, homogeneity_score
import numpy as np

def explore_patterns(X, y_true=None):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    silhouette_avg = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    rand_index = adjusted_rand_score(y_true, labels) if y_true is not None else None
    ch_index = calinski_harabasz_score(X, labels)
    completeness = completeness_score(y_true, labels) if y_true is not None else None
    homogeneity = homogeneity_score(y_true, labels) if y_true is not None else None
    return gmm, silhouette_avg, db_index, rand_index, ch_index, completeness, homogeneity

def select_best_model(models, X_test, y_test):
    best_model = None
    best_score = float('inf')
    best_metrics = {}
    for model in models:
        pred = model.predict(X_test)
        mse_score = mean_squared_error(y_test, pred)
        silhouette_avg = silhouette_score(X_test, pred)
        db_index = davies_bouldin_score(X_test, pred)
        rand_index = adjusted_rand_score(y_test, pred)
        ch_index = calinski_harabasz_score(X_test, pred)
        completeness = completeness_score(y_test, pred)
        homogeneity = homogeneity_score(y_test, pred)
        # Calculate a combined score based on various metrics
        score = mse_score + silhouette_avg + db_index - rand_index + ch_index + completeness + homogeneity
        if score < best_score:
            best_score = score
            best_model = model
            best_metrics = {'MSE': mse_score, 'Silhouette Score': silhouette_avg, 
                            'Davies-Bouldin Index': db_index, 'Adjusted Rand Index': rand_index,
                            'Calinski-Harabasz Index': ch_index, 'Completeness Score': completeness,
                            'Homogeneity Score': homogeneity}
    return best_model, best_metrics
