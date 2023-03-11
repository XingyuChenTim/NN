import numpy as np


def sigmoid(value, dS=False):
    if (dS == True):
        return value * (1 - value)
    return 1 / (1 + np.exp(-value))


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
X_cols = 2
hiddenUnits = 3
LossList = []
W1_old = []
W2_old = []
Out = 1

W1 = np.random.rand(X_cols, hiddenUnits)
print(W1.shape)
print(W1)
W2 = np.random.rand(hiddenUnits, Out)
print(W2.shape)

Z1 = X @ W1
print(Z1)

H = sigmoid(Z1)
print(H)
print(H.shape)

Z2 = H @ W2
print(Z2)
print(Z2.shape)

Yhat = sigmoid(Z2)
print(Yhat)

output_error = Yhat - Y
print(output_error)

Loss = np.sum(output_error ** 2)
print(Loss)
LossList.append(Loss)

YhatD_Error = output_error * sigmoid(Yhat, dS=True)
print(YhatD_Error)
print(YhatD_Error.shape)

print(W2)
YhatD_W2 = YhatD_Error.dot(W2.T)
print(YhatD_W2)
