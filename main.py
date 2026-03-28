import numpy as np
class LogisticRegression:

    def __init__(self, X, y, iters=5000, alpha=0.1, lambda_=1):
        # standardisation
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1

        self.x_train = (X - self.mean) / self.std
        self.y_train = y
        self.initial_bias = 0
        self.initial_weight = np.zeros(self.x_train.shape[1])

        self.iters = iters
        self.alpha = alpha
        self.lambda_ = lambda_
        self.optimal_weight = None
        self.optimal_bias = None


    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


    def cost_function(self, weight, bias):
        # z contains an array of all the z for each records (m examples)
        z = np.dot(self.x_train, weight) + bias
        m = self.x_train.shape[0]
        loss = 0

        for i in range(m):
            predictions = self.sigmoid(z[i])
            predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
            loss += self.y_train[i] * np.log(predictions) + (1 - self.y_train[i]) * np.log(1 - predictions)

        cost = (loss) / (-1 * m)
        return cost


    def regularised_cost_function(self, weight, bias):
        m = self.x_train.shape[0]
        sum_of_weight_squares = 0
        cost_function = self.cost_function(weight, bias)
        for i in range(weight.shape[0]):
            sum_of_weight_squares += weight[i] ** 2

        reg_cost_function = (self.lambda_) / (2 * m) * sum_of_weight_squares
        total_cost = cost_function + reg_cost_function
        return total_cost


    def compute_gradient(self, weight, bias):
        z = np.dot(self.x_train, weight) + bias
        m, n = self.x_train.shape
        dj_dw = np.zeros(weight.shape)
        dj_db = 0

        for i in range(m):
            error = self.sigmoid(z[i]) - self.y_train[i]
            dj_db += error
            for j in range(n):
                dj_dw[j] += error * self.x_train[i, j]

        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db


    def compute_gradient_regularisation(self, weight, bias):
        m = self.x_train.shape[0]
        dj_dw, dj_db = self.compute_gradient(weight, bias)
        dj_dw += (self.lambda_) / (m) * weight
        return dj_dw, dj_db


    def run_gradient_descent(self, weight, bias):
        previous_cost = float("inf")
        for i in range(1, self.iters + 1):
            gradient_w, gradient_b = self.compute_gradient_regularisation(weight, bias)
            weight = weight - self.alpha * gradient_w
            bias = bias - self.alpha * gradient_b
            current_cost = self.regularised_cost_function(weight, bias)

            if i % 100 == 0:
                print(f"Iteration {i}: {current_cost}")

            if abs(previous_cost - current_cost) < 0.00001:
                print(f"Converged at iteration {i}, cost = {previous_cost}")
                print("")
                break

            previous_cost = current_cost

        self.optimal_weight, self.optimal_bias = weight, bias
        return weight, bias


    def fit(self):
        self.run_gradient_descent(self.initial_weight, self.initial_bias)


    def predict(self, x_input):
        # scaling test values
        scaled_input = (x_input - self.mean) / self.std
        z_values = np.dot(scaled_input, self.optimal_weight) + self.optimal_bias
        probabilities = self.sigmoid(z_values)
        return (probabilities >= 0.5).astype(float)



    def mean_accuracy(self, predictions, y):
        return np.mean(predictions == y)



