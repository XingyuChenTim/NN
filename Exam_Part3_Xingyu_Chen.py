# Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# (Satisfied) Overview: You will be writing code for a one hidden-layer, multiple-output NN using Python.
# You will not use any NN packages

# (Satisfied) Your code accepts labeled data of size n by c. The first column will be the label.
# The label will be numeric and can be between 2 through k.
# For example, if k is 3, there are three categories, 1, 2, and 3.
# For simplicity, you can assume that k < 5.

# (Satisfied) It is required that your code specifically reads a datafile called “TheData.csv”

# (Satisfied) The code will use EITHER the Sigmoid or ReLU for the hidden layer and the USER will decide.

# (Satisfied) The code will use the Softmax activation for the output layer,
# along with the one-hot-encoding for the Y, and the categorical cross entropy for the Loss.

# (Satisfied) The USER will choose how many epochs to run.

# (Satisfied) The code should show (as output) the total loss, the loss as it goes,
# and a visualization that shows how the loss is decreasing.
# The code should show the final prediction (after softmax and then argmax are applied).
# For example, if Y is [ 1 3  1  2  3]  then the predicted results might be
# (after softmax and then argmax)  [ 1  3  2  2  3].
# Create a confusion matrix to compare.

# (Satisfied) You may choose to use sums or means or a blend in your code – that’s up to you.

# (Partial Satisfied) Enable your code to train in batches and have the USER choose the number of batches.
# I will limit this choice to between 1 batch (the entire dataset at once) to 3 batches.
# This way, the size of the dataset will not be an issue. Remember that training in batches will use sums (or means).

# (Satisfied) Extra Credit (5 points) Your code should split the dataset into testing and training datasets.
# It will train the NN and then test the accuracy using the testing set.
# Create a confusion matrix and note the final accuracy.


# Ask for user input
epoch = eval(input("How many epochs to run (Suggestion: start with 100/500/1000): \n"))
activation_func = eval(input("Choose your activation function for hidden layer \n"
                             "Enter 0 for Sigmoid or Enter 1 for ReLU: \n"))
batches = eval(input("How many batches to run (Enter 1, 2, or 3): \n"))

# read dataset
df = pd.read_csv("TheData.csv")

# Set up variables
InputColumns = len(df.columns) - 1
NumberOfLabels = len(df.iloc[:, 0].unique())
LR = .01
LRB = .01


def split_dataset(sets, labels):
    """
    Split the dataset randomly into 80% training and 20% development set
    """
    indices = list(range(len(sets)))
    num_training_indices = int(0.8 * len(sets))
    np.random.shuffle(indices)
    train_indices = indices[:num_training_indices]
    dev_indices = indices[num_training_indices:]
    # split the actual data
    train_set, train_labels = sets.iloc[train_indices], labels.iloc[train_indices]
    dev_set, dev_labels = sets.iloc[dev_indices], labels.iloc[dev_indices]
    return train_set, train_labels, dev_set, dev_labels


# Split the dataset
train_set, train_labels, dev_set, dev_labels = split_dataset(df.iloc[:, 1:], df.iloc[:, 0])

dev_labels = np.array(dev_labels).T
y_dev_actual = dev_labels
dev_labels = np.array([dev_labels]).T


def mini_batch_gradient_descent(train_set, train_labels, batches):
    indices = list(range(len(train_set)))
    num_training_indices = int(len(train_set)/batches)
    np.random.shuffle(indices)
    train_indices = indices[:num_training_indices]
    result_set, result_labels = train_set.iloc[train_indices], train_labels.iloc[train_indices]
    return result_set, result_labels


train_set, train_labels = mini_batch_gradient_descent(train_set, train_labels, batches)
n = len(train_set)

train_labels = np.array(train_labels).T
y_train_actual = train_labels
train_labels = np.array([train_labels]).T


# Creating one hot labels for y ------------------
def one_hot_labels(y):
    temp = y
    one_hot_labels = np.zeros((n, NumberOfLabels))
    print(one_hot_labels)
    for i in range(n):
        one_hot_labels[i, temp[i] - 1] = 1
    y = one_hot_labels
    return y


train_labels = one_hot_labels(train_labels)


# NN with softmax, hot coding, cross entropy
class NeuralNetwork(object):
    def __init__(self):
        self.InputNumColumns = InputColumns  # columns
        self.OutputSize = NumberOfLabels  # Categories
        self.HiddenUnits = 3  # one layer with h units
        self.n = n  # number of training examples, n
        self.activation_func = activation_func

        print("Initialize NN: \n")
        # Random W1
        self.W1 = np.random.randn(self.InputNumColumns, self.HiddenUnits)  # c by h
        print("INIT W1 is\n", self.W1)
        # Random W2
        self.W2 = np.random.randn(self.HiddenUnits, self.OutputSize)  # h by o
        print("INIT W2 is:\n", self.W2)
        # Random bias for hidden layer
        self.b = np.random.randn(1, self.HiddenUnits)
        print("INIT b's are:\n", self.b)
        # Random bias for last layer
        self.c = np.random.randn(1, self.OutputSize)
        print("INIT c is\n", self.c)

    def FeedForward(self, X):
        print("FeedForward\n\n")
        self.z = (np.dot(X, self.W1)) + self.b
        # X is n by c   W1  is c by h -->  n by h
        print("Z1 is:\n", self.z)

        if self.activation_func:
            self.h = self.ReLU(self.z)  # activation function    shape: n by h
            print("H is:\n", self.h)
        else:
            self.h = self.Sigmoid(self.z)  # activation function    shape: n by h
            print("H is:\n", self.h)

        self.z2 = (np.dot(self.h, self.W2)) + self.c  # n by h  @  h by o  -->  n by o
        print("Z2 is:\n", self.z2)

        # Using Softmax for the output activation
        output = self.Softmax(self.z2)
        return output

    def Sigmoid(self, s, deriv=False):
        if not deriv:
            return 1 / (1 + np.exp(-s))
        return s * (1 - s)

    def Softmax(self, M):
        expM = np.exp(M)
        SM = expM / np.sum(expM, axis=1)[:, None]
        return SM

    def ReLU(self, re, deriv=False):
        if deriv:
            return 1. * (re > 0)
        return np.maximum(0, re)

    def BackProp(self, X, y, output):
        print("\n\nBackProp\n")
        self.LR = LR
        self.LRB = LRB

        # Y^ - Y
        self.output_error = output - y
        print("Y^ - Y\n", self.output_error)

        # NOTE TO READER........................
        # Here - we DO NOT multiply by derivative of Sig for y^ b/c we are using
        # cross entropy and softmax for the loss and last activation
        # REMOVED # self.output_delta = self.output_error * self.Sigmoid(output, deriv=True)
        # So the above line is commented out...............

        self.output_delta = self.output_error

        # (Y^ - Y)(W2)
        self.D_Error_W2 = self.output_delta.dot(self.W2.T)  # D_Error times W2

        # (H)(1 - H) (Y^ - Y)(Y^)(1-Y^)(W2)
        # We still use the Sigmoid on H
        if self.activation_func:
            self.H_D_Error_W2 = self.D_Error_W2 * self.ReLU(self.h, deriv=True)
        else:
            self.H_D_Error_W2 = self.D_Error_W2 * self.Sigmoid(self.h, deriv=True)

        # Note that * will multiply respective values together in each matrix
        # print("Derivative sig H is:\n", self.Sigmoid(self.h, deriv=True))
        # print("self.H_D_Error_W2 is\n", self.H_D_Error_W2)

        ########------UPDATE weights and biases ------------------

        #  XT  (H)(1 - H) (Y^ - Y)(Y^)(1-Y^)(W2)
        self.X_H_D_Error_W2 = X.T.dot(self.H_D_Error_W2)  # this is dW1

        # (H)T (Y^ - Y) -
        self.h_output_delta = self.h.T.dot(self.output_delta)  # this is for dW2

        print("Using sum gradient........\n")
        self.W1 = self.W1 - self.LR * (self.X_H_D_Error_W2)  # c by h  adjusting first set (input -> hidden) weights
        self.W2 = self.W2 - self.LR * (self.h_output_delta)

        print("The sum of the b update is\n", np.sum(self.H_D_Error_W2, axis=0))
        print("The b biases before the update are:\n", self.b)
        self.b = self.b - self.LRB * np.sum(self.H_D_Error_W2, axis=0)
        print("Updated bs are:\n", self.b)

        self.c = self.c - self.LR * np.sum(self.output_delta, axis=0)

        print("The W1 is: \n", self.W1)
        print("The W1 gradient is: \n", self.X_H_D_Error_W2)
        print("The W2 gradient  is: \n", self.h_output_delta)
        print("The biases b gradient is:\n", np.sum(self.H_D_Error_W2, axis=0))
        print("The bias c gradient is: \n", np.sum(self.output_delta, axis=0))
        ################################

    def TrainNetwork(self, X, y):
        output = self.FeedForward(X)
        print("Output in TNN\n", output)
        self.BackProp(X, y, output)
        return output

    def predict(self, X):
        output = self.FeedForward(X)
        return np.argmax(output, axis=1) + 1


MyNN = NeuralNetwork()
TotalLoss = []

for i in range(epoch):
    print("\nRUN:\n ", i)
    output = MyNN.TrainNetwork(train_set, train_labels)

    print("The output is: \n", output)
    MaxValueIndex = np.argmax(output, axis=1)
    print('Prediction y^ is \n', MaxValueIndex + 1)
    df_confusion = pd.crosstab(y_train_actual, MaxValueIndex + 1)
    print("The confusion matrix between actual label and predict label (training set):\n")
    print("Row is actual, Col is predicted:")
    print(df_confusion)
    # Using Categorical Cross Entropy...........
    loss = np.mean(-train_labels * np.log(output))  # We need y to place the "1" in the right place
    print("The current average loss is\n", loss)
    TotalLoss.append(loss)

predicted = MyNN.predict(dev_set)
print("The actual label: ")
print(y_dev_actual)
print("The predict label: ")
print(predicted)
df_confusion = pd.crosstab(y_dev_actual, predicted)
print("The confusion matrix between actual label and predict label (dev set):\n")
print("Row is actual, Col is predicted:")
print(df_confusion)


def accuracy(y_true, y_pred, normalize=True):
    accuracy = []
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    if normalize == True:
        return np.mean(accuracy)
    if normalize == False:
        return sum(accuracy)


print("The accuracy score is : ")
print(accuracy(y_dev_actual, predicted, True))

# Plot total loss
fig1 = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, epoch)
ax.plot(x, TotalLoss)
plt.show()
