import numpy as np
from scipy.stats import mode
"""
Function KnClassification(K, X_train, y_train, X_test):
    Convert X_train, y_train, X_test to NumPy arrays.
    Initialize an array 'predictions' of size equal to the number of test samples.

    For each test sample i in X_test:
        -Compute the Euclidean distance between the i-th test sample and all training samples:
       
        -Sort the distances in ascending order and get the indices:
      
        -Select the first K indices as the nearest neighbors:
  
        -Identify the distance of the K-th nearest neighbor:

       -Initialize an empty list 'tied_neighbors'.
           Add the labels of the first K nearest neighbors to 'tied_neighbors':
       
        -Check for ties:
           For j from K to length of distances:
               If distances in sorted list equals kth_neighbor:
                   Add this to 'tied_neighbors'
               Else:
                   Break the loop.

        -Determine the mode of 'tied_neighbors' (most frequent class label).
    -Return predictions

"""
def KnClassification(K, X_train, y_train, X_test):
    """
    K-Nearest Neighbors Classification from scratch.

    Parameters:
    - K: Number of neighbors to consider.
    - X: Training data (features).
    - y: Training data (labels).
    - X_test: Test data to classify.

    Returns:
    - predictions: The predicted class labels for each test instance.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    n_test = X_test.shape[0]
    predictions_first = np.zeros(n_test)

    for i in range(n_test):
        #Euclidean distances between the i-th test sample and all training samples
        distances = np.sqrt(np.sum((X_train - X_test[i, :]) ** 2, axis=1))

        #the indices of the closest neighbors
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:K]
        kth_neighbor = distances[k_nearest_indices[-1]]

        #Collecting the K nearest neighbors
        tied_neighbors = []
        tied_neighbors.extend(y_train[k_nearest_indices])
        
        for j in range(K, len(distances)):
            if distances[sorted_indices[j]] == kth_neighbor:
                tied_neighbors.append(y_train[sorted_indices[j]])
            else:
                break
        predictions_first[i] = mode(tied_neighbors).mode
    return predictions_first