

## NN - FF and BP
## Gates

#############################################
## Here - there are many versions of X for
## trying different things. You can make X
## whatever YOU want it to be. 
## 
## If/when you update X, you must also update Y
## sometimes the W's, etc. 
## Always think about what you are doing :)
############################################

## Compare this to what you do by hand for small
## examples.

################### IMPORTANT #####################
##
##       If you use different X and y values
##       you must also update the following below
        #     self.InputNumColumns = ??  ## columns
        #     self.OutputSize = ??
        #     self.HiddenUnits = ??  ## one layer with h units
        #     self.n = ?? ## number of training examples, n
        
## ###################################################     

import numpy as np
import pandas as pd


############## Using a dataset with THREE label categories, 1, 2, and 3---------------
filename="HeartRisk_JustNums_3_labels2.csv"
DF = pd.read_csv(str(filename))
print(DF)

InputColumns = 3
NumberOfLabels = 3
n = len(DF) ## number of rows of entire X
## Take the label off of X and make it a numpy array
X = np.array(DF.iloc[:, [1, 2, 3]])
## Set y to the label. Check the shape!
y = np.array(DF.iloc[:,0]).T
y = np.array([y]).T
print("y is\n", y)
LR=.01
LRB = .01
#................................................

###################### Creating one hot labels for y ------------------
temp = y
#print(temp)
one_hot_labels = np.zeros((n, NumberOfLabels))
print(one_hot_labels)
for i in range(n):
    one_hot_labels[i, temp[i]-1] = 1    
#print(one_hot_labels)
y = one_hot_labels
print(y)
##################------------------------------------
  

class NeuralNetwork(object):
    def __init__(self):
        
        self.InputNumColumns = InputColumns  ## columns
        self.OutputSize = 3 ## Categories
        self.HiddenUnits = 2  ## one layer with h units
        self.n = n  ## number of training examples, n
        
        print("Initialize NN\n")
        #Random W1
        self.W1 = np.random.randn(self.InputNumColumns, self.HiddenUnits) # c by h  
       
        print("INIT W1 is\n", self.W1)
        
        ##-----------------------------------------
        ## NOTE ##
        ##
        ## The following are all random. However, you can comment this out
        ## and can set any weights and biases by hand , etc.
        ##
        ##---------------------------------------------
        
        self.W2 = np.random.randn(self.HiddenUnits, self.OutputSize) # h by o 
        print("W2 is:\n", self.W2)
        
        self.b = np.random.randn(1, self.HiddenUnits)
        print("The b's are:\n", self.b)
        ## biases for layer 1
        
        self.c = np.random.randn(1, self.OutputSize)
        print("The c is\n", self.c)
        ## bias for last layer
        
        
    def FeedForward(self, X):
        print("FeedForward\n\n")
        self.z = (np.dot(X, self.W1)) + self.b 
        #X is n by c   W1  is c by h -->  n by h
        print("Z1 is:\n", self.z)
        
        self.h = self.Sigmoid(self.z) #activation function    shape: n by h
        print("H is:\n", self.h)
        
        self.z2 = (np.dot(self.h, self.W2)) + self.c # n by h  @  h by o  -->  n by o  
        print("Z2 is:\n", self.z2)
        
        ## Using Softmax for the output activation
        output = self.Softmax(self.z2)  
        print("output Y^ is:\n", output)
        return output
        
    def Sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))
    
    def Softmax(self, M):
        #print("M is\n", M)
        expM = np.exp(M)
        #print("expM is\n", expM)
        SM=expM/np.sum(expM, axis=1)[:,None]
        #print("SM is\n",SM )
        return SM 
    
    def BackProp(self, X, y, output):
        print("\n\nBackProp\n")
        self.LR = LR
        self.LRB=LRB  ## LR for biases
        
        # Y^ - Y
        self.output_error = output - y    
        print("Y^ - Y\n", self.output_error)
        
        ## NOTE TO READER........................
        ## Here - we DO NOT multiply by derivative of Sig for y^ b/c we are using 
        ## cross entropy and softmax for the loss and last activation
        # REMOVED # self.output_delta = self.output_error * self.Sigmoid(output, deriv=True) 
        ## So the above line is commented out...............
        
        self.output_delta = self.output_error 
          
        ##(Y^ - Y)(W2)
        self.D_Error_W2 = self.output_delta.dot(self.W2.T) #  D_Error times W2
        #print("W2 is\n", self.W2)
        #print(" D_Error times W2\n", self.D_Error_W2)
        
        ## (H)(1 - H) (Y^ - Y)(Y^)(1-Y^)(W2)
        ## We still use the Sigmoid on H
        
        self.H_D_Error_W2 = self.D_Error_W2 * self.Sigmoid(self.h, deriv=True) 
        
        ## Note that * will multiply respective values together in each matrix
        #print("Derivative sig H is:\n", self.Sigmoid(self.h, deriv=True))
        #print("self.H_D_Error_W2 is\n", self.H_D_Error_W2)
        
        ################------UPDATE weights and biases ------------------
        #print("Old W1: \n", self.W1)
        #print("Old W2 is:\n", self.W2)
        #print("X transpose is\n", X.T)
        
        ##  XT  (H)(1 - H) (Y^ - Y)(Y^)(1-Y^)(W2)
        self.X_H_D_Error_W2 = X.T.dot(self.H_D_Error_W2) ## this is dW1
        
        ## (H)T (Y^ - Y) - 
        self.h_output_delta = self.h.T.dot(self.output_delta) ## this is for dW2
        
        #print("the gradient :\n", self.X_H_D_Error_W2)
        #print("the gradient average:\n", self.X_H_D_Error_W2/self.n)
        
        print("Using sum gradient........\n")
        self.W1 = self.W1 - self.LR*(self.X_H_D_Error_W2) # c by h  adjusting first set (input -> hidden) weights
        self.W2 = self.W2 - self.LR*(self.h_output_delta) 
        
        
        print("The sum of the b update is\n", np.sum(self.H_D_Error_W2, axis=0))
        print("The b biases before the update are:\n", self.b)
        self.b = self.b  - self.LRB*np.sum(self.H_D_Error_W2, axis=0)
        #print("The H_D_Error_W2 is...\n", self.H_D_Error_W2)
        print("Updated bs are:\n", self.b)
        
        self.c = self.c - self.LR*np.sum(self.output_delta, axis=0)
        #print("Updated c's are:\n", self.c)
        
        print("The W1 is: \n", self.W1)
        print("The W1 gradient is: \n", self.X_H_D_Error_W2)
        #print("The W1 gradient average is: \n", self.X_H_D_Error_W2/self.n)
        print("The W2 gradient  is: \n", self.h_output_delta)
        #print("The W2 gradient average is: \n", self.h_output_delta/self.n)
        print("The biases b gradient is:\n",np.sum(self.H_D_Error_W2, axis=0 ))
        print("The bias c gradient is: \n", np.sum(self.output_delta, axis=0))
        ################################################################
        
    def TrainNetwork(self, X, y):
        output = self.FeedForward(X)
        print("Output in TNN\n", output)
        self.BackProp(X, y, output)
        return output

#-------------------------------------------------------------------        
MyNN = NeuralNetwork()

TotalLoss=[]
AvgLoss=[]
Epochs=1000

for i in range(Epochs): 
    print("\nRUN:\n ", i)
    output=MyNN.TrainNetwork(X, y)
   
    #print("The y is ...\n", y)
    print("The output is: \n", output)
    MaxValueIndex=np.argmax(output, axis=1)
    print('Prediction y^ is', MaxValueIndex+1)
    ## Using Categorical Cross Entropy...........
    loss = np.mean(-y * np.log(output))  ## We need y to place the "1" in the right place
    print("The current average loss is\n", loss)
    TotalLoss.append(loss)
    AvgLoss.append(loss)
    
    ## OLD---------------------
    # OLD #print("Total Loss:", .5*(np.sum(np.square(output-y))))
    # OLD #TotalLoss.append( .5*(np.sum(np.square(output-y))))
    #print("Average Loss:", .5*(np.mean(np.square((output-y)))))
    #AvgLoss.append(.5*(np.mean(np.square((output-y)))))
   
    
###################-output and vis----------------------    
#print("Total Loss List:", TotalLoss) 
import matplotlib.pyplot as plt

fig1 = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, Epochs)
ax.plot(x, TotalLoss)    

# AvgLoss_ = np.mean(AvgLoss)
# print(AvgLoss_)
# fig2 = plt.figure()
# ax = plt.axes()
# x = np.linspace(0, 10, Epochs)
# ax.plot(x, AvgLoss_)  

print(y)

