import numpy as np
X = np.array([[0,0], [0, 1],[1,0], [1,1]])   ## 4 by 2
## Inputs are n rows by c features/columns
print(X)
print(X.shape)
Y = np.array([[0],[1],[1],[0]])
numOutputs = 1
NumHidden = 2
##-------- INPUT LAYES  X --> H
W1=np.array([[1,1], [1,1]]) ## shape is c by h
print(W1.shape)
bs=np.array([[0, -1]]) ## shape 1 by h
print(bs.shape)
Z1 = (X@W1) + bs  ## n by c @  c by h -> n by h
print(Z1.shape)
A_Z1  = np.maximum(0,Z1)  ## n by h
print(A_Z1.shape)
##-----------------HIDDEN to OUTPUT
W2= np.array([[1],[-2]]) # shape h by 1
print(W2.shape)
c= 0
Z2 = A_Z1@W2 + c ## shape n by h @ h by 1
y_hat=np.maximum(0, Z2) ## shape n by 1
print(y_hat)
