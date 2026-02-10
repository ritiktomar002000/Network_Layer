import numpy as np
import pickle


class NeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.0075, num_iterations=2500):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = {}

    def initialize_parameters(self):
        np.random.seed(1)
        for l in range(1, len(self.layer_dims)):
            self.parameters[f"W{l}"] = (
                np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01
            )
            self.parameters[f"b{l}"] = np.zeros((self.layer_dims[l], 1))

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_backward(dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def forward_propagation(self, X):
        caches = []
        A = X

        for l in range(1, len(self.layer_dims) - 1):
            Z = np.dot(self.parameters[f"W{l}"], A) + self.parameters[f"b{l}"]
            A = self.relu(Z)
            caches.append((A, Z))

        ZL = np.dot(self.parameters[f"W{len(self.layer_dims) - 1}"], A) + \
             self.parameters[f"b{len(self.layer_dims) - 1}"]
        AL = self.sigmoid(ZL)
        caches.append((AL, ZL))

        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        return cost

    def backward_propagation(self, X, Y, caches):
        grads = {}
        m = X.shape[1]
        L = len(self.layer_dims) - 1

        AL, ZL = caches[-1]
        dZL = AL - Y
        grads[f"dW{L}"] = (1 / m) * np.dot(dZL, caches[-2][0].T)
        grads[f"db{L}"] = (1 / m) * np.sum(dZL, axis=1, keepdims=True)

        dA = np.dot(self.parameters[f"W{L}"].T, dZL)

        for l in reversed(range(1, L)):
            A_prev, Z = caches[l - 1]
            dZ = self.relu_backward(dA, Z)
            A_prev_input = X if l == 1 else caches[l - 2][0]
            grads[f"dW{l}"] = (1 / m) * np.dot(dZ, A_prev_input.T)
            grads[f"db{l}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.parameters[f"W{l}"].T, dZ)

        return grads

    def update_parameters(self, grads):
        for l in range(1, len(self.layer_dims)):
            self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

    def fit(self, X, Y):
        self.initialize_parameters()

        for i in range(self.num_iterations):
            AL, caches = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(X, Y, caches)
            self.update_parameters(grads)

            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return (AL > 0.5).astype(int)


def load_data(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)


def main():
    data = load_data("data/data_dog_nondog.pickle")

    X_train, Y_train = data["X_train"], data["Y_train"]
    X_test, Y_test = data["X_test"], data["Y_test"]

    nn = NeuralNetwork(layer_dims=[X_train.shape[0], 20, 7, 1])
    nn.fit(X_train, Y_train)

    train_acc = 100 - np.mean(np.abs(nn.predict(X_train) - Y_train)) * 100
    test_acc = 100 - np.mean(np.abs(nn.predict(X_test) - Y_test)) * 100

    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
