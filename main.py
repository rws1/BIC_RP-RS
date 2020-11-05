import pandas as pd
import numpy as np
import time


# sigmoid value of input X and associated weight
# dot function implements matrix multiplication
def sigmoid(X, weight):
    z = np.dot(X, weight)
    return 1 / (1 + np.exp(-z))

# this is cross entropy loss where h is the predicted value and y is the actual value
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]


def update_weight_loss(weight, learning_rate, gradient):
    return weight - learning_rate * gradient


data = pd.read_csv("/home/rachana/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Dataset size")
print("Rows {} Columns {}".format(data.shape[0], data.shape[1]))

df = data.copy()

df['class'] = df['Churn'].apply(lambda x: 1 if x == "Yes" else 0)
# features will be saved as X and our target will be saved as y
X = df[['tenure', 'MonthlyCharges']].copy()

y = df['class'].copy()
start_time = time.time()

num_iter = 100000

intercept = np.ones((X.shape[0], 1))
X = np.concatenate((intercept, X), axis=1)
theta = np.zeros(X.shape[1])

for i in range(num_iter):
    h = sigmoid(X, theta)
    gradient = gradient_descent(X, h, y)
    theta = update_weight_loss(theta, 0.1, gradient)

print("Training time (Log Reg using Gradient descent):" + str(time.time() - start_time) + " seconds")
print("Learning rate: {}\nIteration: {}".format(0.1, num_iter))

result = sigmoid(X, theta)
f = pd.DataFrame(np.around(result, decimals=6)).join(y)
f['pred'] = f[0].apply(lambda x: 0 if x < 0.5 else 1)
print("Accuracy (Loss minimization):")
print(f.loc[f['pred'] == f['class']].shape[0] / f.shape[0] * 100)
