import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy


class ANNClassifier:
    """
        Simple implementation ANN with 2 layer (1 hidden layer, 1 output layer).
        This implementation works for binary classification problem.
    """

    def __init__(self, layers=[13, 8, 1], lr=0.001, epochs=800):
        self.lr = lr
        self.epochs = epochs
        self.layers = layers
        self.parameter = {}

    def generate_random_weights_bias(self):
        np.random.seed(8)
        self.parameter['W1'] = np.random.randn(self.layers[0], self.layers[1])
        self.parameter['b1'] = np.random.randn(self.layers[1],)
        self.parameter['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.parameter['b2'] = np.random.randn(self.layers[2],)

    def activation_function(self, input_val):
        """
            This implementation use ReLU (Rectified Linear Unit) as activation function.
        """
        return np.maximum(0, input_val)

    def sigmoid(self, input_val):
        return 1 / (1 + np.exp(-input_val))

    def loss_function(self, y, y_hat):
        return np.mean((y_hat - y)**2)

    def forward_propagation(self, X, y):
        """
            Implement forward propagation, simply will sum and pass to activation function
            until output value.
        """

        weighted_sum_1 = np.dot(X, self.parameter['W1']) + self.parameter['b1']
        activation_result_1 = self.activation_function(weighted_sum_1)

        # Because this is output layer activation function will change using sigmoid
        weighted_sum_2 = np.dot(
            activation_result_1,
            self.parameter['W2']
        ) + self.parameter['b2']
        activation_result_2 = self.sigmoid(weighted_sum_2)

        y_hat = activation_result_2
        loss = self.loss_function(y, y_hat)

        # Save activation result and weighted sum value
        self.parameter['WS1'] = weighted_sum_1
        self.parameter['AR1'] = activation_result_1
        self.parameter['WS2'] = weighted_sum_2
        self.parameter['AR2'] = activation_result_2

        return y_hat, loss

    def dRelu(self, X: np.array):
        new_X = copy.deepcopy(X)
        new_X[new_X >= 0] = 1
        new_X[new_X < 0] = 0

        return new_X

    def fit(self, X, y):
        self.generate_random_weights_bias()

        # Stochastic gradient descent
        for _ in range(self.epochs):
            y_hat, _ = self.forward_propagation(X, y)

            # Implement back propagation.
            # d_something means derivative loss with respect to something.
            # np.maximum(0.0000001, something) just make sure not pass 0 value
            # to function that sensitive with 0 value.
            not_zero_yhat = np.maximum(0.0001, y_hat)
            not_zero_yhat_prime = np.maximum(0.0001, 1 - y_hat)
            d_yhat = -np.divide(y, not_zero_yhat) + \
                np.divide(1 - y, not_zero_yhat_prime)
            d_sigmoid = (1 - y_hat) * y_hat
            d_weightedSum2 = d_yhat * d_sigmoid

            d_activationResult1 = np.dot(
                d_weightedSum2, self.parameter['W2'].T)
            d_W2 = np.dot(self.parameter['AR1'].T, d_weightedSum2)
            d_b2 = np.sum(d_weightedSum2, axis=0)

            d_relu = self.dRelu(self.parameter['WS1'])
            d_weightedSum1 = d_activationResult1 * d_relu
            d_W1 = X.T.dot(d_weightedSum1)
            d_b1 = np.sum(d_weightedSum1, axis=0)

            # Update the weight and bias
            self.parameter['W1'] -= self.lr * d_W1
            self.parameter['b1'] -= self.lr * d_b1
            self.parameter['W2'] -= self.lr * d_W2
            self.parameter['b2'] -= self.lr * d_b2

    def predict(self, X):
        ws1 = np.dot(X, self.parameter['W1']) + self.parameter['b1']
        ar1 = self.activation_function(ws1)
        ws2 = np.dot(ar1, self.parameter['W2']) + self.parameter['b2']

        preds = self.sigmoid(ws2).flatten()

        return [1 if pred > 0.5 else 0 for pred in preds]


def main(filename):
    # Import dataset
    df = pd.read_csv(filename)

    # View dataset
    print(df.head(2), '\n')
    print(df.shape, '\n')
    print(df.isna().sum(), '\n')

    X = np.array(df.drop(columns='target'))
    X = StandardScaler().fit_transform(X)
    y = np.array(df[['target']])

    _, num_feature = X.shape
    # Define layers, 8 => optional can be anything, 1 because binary classification.
    ann_layers = [num_feature, 8, 1]

    clf = ANNClassifier(layers=ann_layers, lr=0.0008, epochs=700)
    clf.fit(X, y)

    # Predict
    predict = np.array([
        [43, 1, 0, 150, 247, 0, 1, 171, 0, 1.5, 2, 0, 2],
        [56, 0, 0, 134, 409, 0, 0, 150, 1, 1.9, 1, 2, 3],
        [54, 1, 2, 150, 232, 0, 0, 165, 0, 1.6, 2, 0, 3],
        [50, 1, 0, 144, 200, 0, 0, 126, 1, 0.9, 1, 0, 3],
        [54, 1, 2, 125, 273, 0, 0, 152, 0, 0.5, 0, 1, 2],
        [51, 1, 3, 125, 213, 0, 0, 125, 1, 1.4, 2, 1, 2],
        [46, 0, 2, 142, 177, 0, 0, 160, 1, 1.4, 0, 0, 2]
    ])

    print(clf.predict(predict))


if __name__ == "__main__":
    main()
