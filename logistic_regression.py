#refereneces
#https://medium.com/@koushikkushal95/logistic-regression-from-scratch-dfb8527a4226
#https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
#https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf
#https://www.deeplearningbook.org/
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, n):
        return 1 / (1 + np.exp(-n))

    def compute_cross_entropy(self, y_gtruth, y_prediction):
        epsilon = 1e-9  # Prevent division by zero
        y_1 = y_gtruth * np.log(y_prediction + epsilon)
        y_2 = (1 - y_gtruth) * np.log(1 - y_prediction + epsilon)
        return -np.mean(y_1 + y_2)

    def train(self, X_train, y_train):
        number_of_samples, number_of_features = X_train.shape

        # Initialize weights and bias
        self.weights = np.zeros(number_of_features)
        self.bias = 0

        for _ in range(self.number_of_iterations):
            # Forward pass
            linear_model = np.dot(X_train, self.weights) + self.bias
            forward = self.sigmoid(linear_model)

            # Compute loss
            loss = self.compute_cross_entropy(y_train, forward)
            self.losses.append(loss)

            # Backward pass (gradients)
            dz = forward - y_train
            dw = (1 / number_of_samples) * np.dot(X_train.T, dz)
            db = (1 / number_of_samples) * np.sum(dz)

            # Parameter updates
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X_test):
        threshold = 0.5
        linear_model = np.dot(X_test, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > threshold else 0 for i in y_pred]
        return np.array(y_predicted_class)

    


