# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:56:37 2022

@author: profa
"""

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Read in the data
## Labeled numeric data
## Example:
# Label	  Cholesterol	Weight	Height
# Risk	     251	       267	    70
# NoRisk	105	        103	     62

## DATA HERE:
## https://drive.google.com/file/d/1Ydi5zIyZUbYvCJjfpL2S-EbS9AeWAjOH/view?usp=sharing

filename="HeartRisk_JustNums_Labeled_TwoClasses.csv"

## NOtice that I did not hard-code the filename
HeartHealthDF = pd.read_csv(filename)
## Check to see if this worked
print(type(HeartHealthDF))
print(HeartHealthDF)

## Convert the qualitative labels to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
HeartHealthDF['Label'] = le.fit_transform(HeartHealthDF['Label'])
print(HeartHealthDF)

## Split DF into training and testing data first
train, test = train_test_split(HeartHealthDF, test_size=0.33)
print(train)
print(test)

## Take off the label and make it numeric
## for training adn testing datasets
TrainLabel=train["Label"]
print(TrainLabel)
TestLabel=test["Label"]
print(TestLabel)

## Drop label from datasets
train = train.drop('Label', axis=1)
print(train)

test= test.drop('Label', axis=1)
print(test)




## The fit method requires a numpy array.
## Let's convert
train=train.to_numpy()
print(train)
test=test.to_numpy()
print(test)
TrainLabel=TrainLabel.to_numpy()
print(TrainLabel)
TestLabel=TestLabel.to_numpy()
print(TestLabel)


## Instantiate Perceptron in sklearn
## Parameters and defaults
## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
MyPerceptron = Perceptron(max_iter=1000)

## !! The fit method requires an array or sparse matrix
## not a DF
## Run (fit the data to) the perceptron
TwoD_Train=train[:,(0,2)]
print(TwoD_Train)
MyPerceptron.fit(TwoD_Train, TrainLabel)
#MyPerceptron.fit(TwoD_Train, TrainLabel)

## Test the perceptron
#print(test)
TwoD_Test=test[:,(0,2)]
print(TwoD_Test)
print("\n\nThe predicted labels are:")
print(MyPerceptron.predict(TwoD_Test))

## Print the actual known test labels
print("\nThe true test labels are:")
print(TestLabel)

print(train)
print(train.shape)  ## rows, columns

# weights=np.zeros(1 + train.shape[1])
# print(weights)
# print(weights[1:])

########################################
## For a basic plot in 2D
## for this plot, I will remove one of the dimensions from our
## dataset
########################################

fig = plt.figure(figsize=(10,8))
plt.plot(train[:, 0][TrainLabel == 0], train[:, 1][TrainLabel == 0], 'r^')
plt.plot(train[:, 0][TrainLabel == 1], train[:, 1][TrainLabel == 1], 'bs')
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.title('Classification Data with 2 classes')

############################################
##
## Build our own perceptron
## and
## Visualize the classification
##
###############################################

## constructor method - initialize the object
## Define the class first..............
X = TwoD_Train
print(X)
y = TrainLabel
print(y)

class New_Perceptron(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta #Estimated Time of Arrival
        ## Estimated time before the model finishes
        ## one epoch of training
        self.epochs = epochs
        #An epoch means training the neural network
        #with all the training data for one cycle.
        #In an epoch, we use all of the data exactly once.
        #A forward pass and a backward pass together
        #are counted as one pass.

    def train(self, X, y):
        # w_ are the weights.
        # X is a numpy array/matrix of the data
        ## Create inital weights - one more than you have
        ## columns
        ## w0 + w1x1 + w2x2 + ....
        self.w_ = np.zeros(1 + X.shape[1]) ##
        self.errors_ = []

        for _ in range(self.epochs):
            ## The _ in range runs for self-epochs
            ## but the value does not matter - just the
            ## count.
            errors = 0
            ## X is the dataset as a numpy array
            ## y are the labels of the dataset
            for xi, target in zip(X, y):
                ## first time
                ## update = .01*(label - either 1 or -1)
                ## if the label is 1 and the net input >0
                ## then update will be 0
                ## Otherwise, if the label is 1 and the
                ## net input < 0
                ## Then update will be set to
                ## .01 (1 - - 1) = .02
                update = self.eta * (target - self.predict(xi))
                ## Now update the weights
                self.w_[1:] +=  update * xi  ## This updates the xis
                ## Use the update value to adjust the weights
                ##
                self.w_[0] +=  update  ## This is b the bais
                errors += int(update != 0.0)
                ## keep track of errors which occur when
                ## update is not 0 wich means the label
                ## what not what it should have been
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        ##Inner product
        ## Recall that w_[0] is b the bias
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        ## Here, if the net input (the linear equation)
        ## is >=0, the output is 1, otherwise its -1.
        return np.where(self.net_input(X) >= 0.0, 1, 0)
#-------------------------------------------------------------

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

MyNewPerceptron=New_Perceptron(epochs=60, eta=.1)
MyNewPerceptron.train(X, y)
MyNewPerceptron.predict(TwoD_Test)


#print(train[:,(0,1)])
#print(TrainLabel)
#print('Weights: %s' % MyNewPerceptron.w_)
#print(train[:,(0,1)].shape)
#TL=TrainLabel.transpose()
#print(TL)
#print(TL.shape)
plot_decision_regions(X, y, clf=MyNewPerceptron)
plt.title('Small Perceptron Example for Heart Binary Data')
plt.xlabel('Cholesterol')
plt.ylabel('Weight')
plt.show()

plt.plot(range(1, len(MyNewPerceptron.errors_)+1), MyNewPerceptron.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()

