# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:19:26 2022

@author: profa
"""

##################
 ## NN in Python
 ## Gates
 
 ## DATA IS HERE
 # https://drive.google.com/file/d/19K2pF77RcxlLNvX9E5hLJP7kBuzYiF1-/view?usp=sharing
 
#########################################

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


## Create and Train a TINY NN on easy data
filename="C:/Users/profa/Desktop/GatesBoltonAnalyticsSite/HeartRisk_JustNums_Labeled.csv"

#######SIDE NOTE--------------------------------------------
## If your data is on the web - you can GET it using keras-
##
## csv_file = tf.keras.utils.get_file('filename.csv', 'https://path_to_file_on web.csv')
##--------------------------------------------------------------------


## NOtice that I did not hard-code the filename
HeartHealthDF = pd.read_csv(filename)
## Check to see if this worked
print(type(HeartHealthDF))
print(HeartHealthDF.head(15))

## We have labels - but they are words. 
## We will need to encode them as numbers.

from sklearn.preprocessing import LabelEncoder
Label_Encoder = LabelEncoder() ##instantiate your encoder
HeartHealthDF["Label"] = Label_Encoder.fit_transform(HeartHealthDF["Label"])
#print(HeartHealthDF)

## Now we have this:
#         Label  Cholesterol  Weight  Height
# 0       2          251     267      70
# 1       1          105     103      62
# 2       0          156     193      72
# 3       1          109     100      63

##### NOTE: Here, my label is called Label. However
## that is rarely the case. The label may have been called
## color (if you are predicting a color) or salary level, etc. 
## In other words, it is important to understand the data
## and label that you have.

## From here - one option is to "scale" our data. 
## Let's do that to see how it works. 
## DO NOT SCALE THE LABEL!! Remember - the label is NOT
## your data. It is the category each data row belongs to ;)

from sklearn.preprocessing import StandardScaler
MyScaler = StandardScaler()  #instantiate the scaler
HeartHealthDF[["Cholesterol",  "Weight",  "Height"]] = MyScaler.fit_transform(HeartHealthDF[["Cholesterol",  "Weight",  "Height"]])
#print(HeartHealthDF)

## We are using NN here to TRAIN a model. We will then use
## that model to see if we can predict the right label

## To TRAIN the model, we need a training dataset.
## To TEST the model's accuracy (etc) we need a DISJOINT Testing dataset
## Why disjoint?

from sklearn.model_selection import train_test_split
TRAIN_data, TEST_data = train_test_split(HeartHealthDF, test_size = 0.25)
#print(TRAIN_data)
#print(TEST_data)

## Make sure your training and testing data are balanced.
## In other words, that there is a fair representation 
## of all labels. 

##  IMPORTANT ##
##
## Right now, our testing and training datasets
## still have the labels ON THEM.
## We need to remove the label and save it. 

## Get the label from the training data
Train_Label = TRAIN_data["Label"]  ## Save the label
#print(Train_Label)
## Drop the label from the training set now that you saved it
TRAIN_data = TRAIN_data.drop("Label",  axis=1)
## axis = 1 means drop the column. axis = 0 drops a row
#print(TRAIN_data)

##OK! Let's do this for the testing data now
Test_Label = TEST_data["Label"]  ## Save the label
print(Test_Label)
## Drop the label from the training set now that you saved it
TEST_data = TEST_data.drop("Label",  axis=1)
## axis = 1 means drop the column. axis = 0 drops a row
print(TEST_data)


####################
## Now - what do we have?
##
## We have numeric data. 
## We have three dimensional data (3 variables)
## We split our data into a training set and a testing set
## We have the labels for each SEPERATELY!
###################################################

########################
## Run the NN
##########################################
from sklearn.neural_network import MLPClassifier
## Instantiate your NN with the parameter values you want
MyNN = MLPClassifier(hidden_layer_sizes=(50,80,50), 
                     max_iter=100,activation = 'relu',solver='adam',random_state=1)
## hidden_layer_sizes specifies the number of layers (3 there 
## because we have three values in our tuple. )
## We are also specifying the number of nodes in the hidden layer. 


########## Train the NN Model
MyNN.fit(TRAIN_data, Train_Label)
## Notice that we give the model the data and
## the label for the data seperately!

## Now we can use our test data (WITHOUT the label)
## to see if our model predicts the label. 
## So - the model will predict what it thinks the label
## should be. 
## We have the labels, so we can check to see which labels
## the model predicted right and wrong. 
## We will use a confusion matrix for this

Test_Prediction = MyNN.predict(TEST_data)
#print(Test_Prediction)
#print(Test_Label)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#Comparing the predictions against the actual observations 
MyConfusionMatrix = confusion_matrix(Test_Prediction, Test_Label)
print("\n\n",MyConfusionMatrix)
#Printing the accuracy
print("The accuracy of the sklearn MLPClassifier is:")
print(accuracy_score(Test_Prediction, Test_Label))