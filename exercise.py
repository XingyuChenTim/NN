# -*- coding: utf-8 -*-
"""
author: Xingyu Chen
"""

import numpy as np

X = np.array([[3, 4, 8], [6, -2, -1], [1, 2, 7], [-2, 5, -4], [-5, 9, -6]])
print("The input is:\n", X)
# print(X.shape)  ## 4 by 2   (n by c)
Y = np.array([[0], [1], [0], [1], [0]])
print("The target value are:\n", Y)
# print(Y.shape)

## Size of each input x
X_examples = X.shape[0]  ## each x has 4 rows (examples)
print("The number of input vectors are:\n", X_examples)
X_vectorSize = X.shape[1]

numOutputs = 1
NumHidden = 3

W1_x = np.array([[1, 2, 0, 1], [-1, -2, 1, 0], [0, 1, -1, 2]])
print("The W1 weights for the x's are:\n", W1_x)
# print(W1_x.shape)
## Here, W1_x  is 2 by 3   (c by h)

bs = np.array([[8, 9, 10, 11]])  ## bs shape should be 1 by h
print("The b are: \n", bs)
# print(bs.shape)

Z1 = X @ W1_x + bs  # should be shape n by h
print("The Z1 are: \n", Z1)
# print(Z1.shape)

W2_h = np.array([[1], [-2], [0], [-1]])
# print(W2_h.shape)
print("The W2_h are: \n", W2_h)
##W2_h FROM hidden units into the output.
## The shape must be h by 1
c = 0  ## This is the shape of the output which is 1 by 1

A_Z1 = np.maximum(0, Z1)  ##ReLU,  shape n by h

Z2 = (A_Z1 @ W2_h) + c  ## n by h  @  h by 1
## Z2 shape is n by 1
y_hat = np.maximum(0, Z2)  ## ReLU #shape n by 1
print("The output is:\n", y_hat)
# print(y_hat.shape)
