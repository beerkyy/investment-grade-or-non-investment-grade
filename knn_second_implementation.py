#however skitlearn uses different approach
from scipy.stats import mode
import numpy as np

def KnClassification_v2(K, X_train, y_train, X_test):
    """
    K-Nearest Neighbors Classification from scratch.

    Parameters:
    - K: Number of neighbors to consider.
    - X_train: Training data (features).
    - y_train: Training data (labels).
    - X_test: Test data to classify.

    Returns:
    - predictions: The predicted class labels for each test instance.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    n_test = X_test.shape[0]
    predictions = np.zeros(n_test)

    for i in range(n_test):
        # Euclidean distances between the i-th test sample and all training samples
        distances = np.sqrt(np.sum((X_train - X_test[i, :]) ** 2, axis=1))

        # Indices of the closest neighbors
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:K]

        # Labels for the K nearest neighbors
        k_nearest_labels = y_train[k_nearest_indices]

        #scipy.stats.mode function in newer versions of SciPy implementation, 
        #it was different before but slighly changed it
        predictions[i] = mode(k_nearest_labels, keepdims=False).mode

    return predictions
