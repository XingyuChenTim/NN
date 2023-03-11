# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 13:14:50 2022

@author: profa
"""

#######################################
##
## Neural Network - by hand for XOR
##
## Basic (no back prop)
## One hidden layer
##
## Gates
########################################

import numpy as np

## First - what do we know
## Here, X contains all the x's
X = np.array([[0,0], [0, 1],[1,0], [1,1]])
#print(X.shape)  ## (4, 2) - 4 rows &24 columns
print("The input for XOR is:\n", X)
## This is actually
##   0 0 
##   0 1
##   1 0 
##   1 1
##
## Each column vector is an input
## We have x1, x2, x3, and 4 here.
##
## Keep track of the shapes of your 
## matrices. 

Y = np.array([[0],[1],[1],[0]])
#print(Y.shape) ## (4, 1) - 4 rows, 1 cols
print("The target value for XOR are:\n", Y)

## Size of each input x
X_examples = X.shape[0] ## each x has 4 rows (examples)
print("The number of input vectors are:\n",X_examples)
X_vectorSize=X.shape[1]
#print(X_vectorSize)

## Number of outputs
numOutputs = 1
## Neurons in hidden layer
NumHidden=2
#LearningRate = .01 ## for back prop
## We can set a seed so that each time
## we run, its the same
#np.random.seed(1234)

## Our WEIGHT matrices-----------
## Matrix W1 contains w11, w12, w21, w22
## Here, we can use all 1's or we
## can make it random and then use
## back prop

## We need WTX - let's make sure the shapes are right
## (2,4)  @ (4,2) = (2,2)
## This is the weight matrix for the X 
## Here, I am doing x@W
## If you do W@X you may need to update some shapes
#W1_x = np.random.rand(NumHidden, X_vectorSize) ##for backprop
W1_x=np.array([[1,1],[1,1]])
#print(W1_x.shape)
print("The W1 weights for the x's are:\n",W1_x)
## Bias for x to hidden
#b_x=np.random.randn(W1_x.shape[1], 1) #for back prop
c_x=np.array([[0,-1]])
print("The bias for the x's is:\n",c_x)
#print(c_x.shape)
## Keep and eye on the shape as you will be
## doing W1X

## This is the weight matrix for the H
#W2_h = np.random.rand(numOutputs, NumHidden) #for back prop
W2_h=np.array([[1],[-2]])
#print(W2_h.shape)
print("THe Weight matrix for the h's is:\n", W2_h)
## Here again - I will do H@W2_h
## If you want to multiple the weights first - 
## you may need to make some updates.
## Bias for hidden to output
#b_h=np.random.randn(W2_h.shape[1], 1) #for back prop
b_h=0
#print(b_h.shape)

##Keep track of losses
LossList=[]

# Forward propagation as a function
def FeedForward(w1,w2,x, c, b):
    X_to_H1 = np.dot(x, w1) 
    print("The XW is:\n", X_to_H1)
    X_to_H1 = X_to_H1 + c
    print("The XW+c is:\n", X_to_H1)
    #print(X_to_H1.shape)
    #a1 = sigmoid(X_to_H1)    
    H_out=np.maximum(0,X_to_H1) ## ReLU
    #print(H_out)
    H_to_output = np.dot(H_out,w2) + b_h
    #print(H_to_output)
    Output = np.maximum(0,H_to_output) ## ReLU
    #print(Output)
    return X_to_H1,H_out,H_to_output,Output


IN,H,Hout,OUT=FeedForward(W1_x, W2_h, X, c_x, b_h)
print("The output is:\n", OUT)





