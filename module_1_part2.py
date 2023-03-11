# # Assignment Description
# 1) Create a numeric dataset with 3 rows, 3 columns/variables, and a 0 or 1 label. Be sure to CREATE it :)
#
# For example - I created this one:
#
# Label    Cholesterol    Weight    Height
# 1              251             267           70
# 0             105              103           62
# 1              196             193           72
#
# Next, choose weights and an activation function.
#
# Then, illustrate by hand how row 1 would "push forward" through your perceptron.

# import library
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {'Label': [1, 0, 0],
        'Cholesterol': [221, 186, 170],
        'Weight': [240, 172, 129],
        'Height': [69, 72, 66]}
# Read Data
df = pd.DataFrame(data)

print(df)

# Split DF into training and testing data first
train, test = train_test_split(df, test_size=0.33)

# Take off the label and make it numeric
# for training adn testing datasets
TrainLabel = train["Label"]
TestLabel = test["Label"]

# Drop label from datasets
train = train.drop('Label', axis=1)
test = test.drop('Label', axis=1)

# The fit method requires a numpy array.
# Let's convert
train = train.to_numpy()
print('Train set w/o label:\n', train)
test = test.to_numpy()
print('Test set w/o label:\n', test)
TrainLabel = TrainLabel.to_numpy()
print('Train set label:\n', TrainLabel)
TestLabel = TestLabel.to_numpy()
print('Test set label:\n', TestLabel)

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
                return np.dot(X, self.w_[1:]) + self.w_[0]

        def Sig(self, Y):
                return 1 / (1 + np.exp(-Y))

        def predict(self, Y):
                return np.where(self.Sig(self.net_input(Y)) >= 0.0, 1, 0)
#-------------------------------------------------------------

MyNewPerceptron=New_Perceptron(epochs=20, eta=.1)
MyNewPerceptron.train(train, TrainLabel)

print('Weights: %s' % MyNewPerceptron.w_)
print('Errors: %s' % MyNewPerceptron.errors_)

print("\nThe perceptrons predict test labels are:")
print(MyNewPerceptron.predict(test))

print("\nThe true test labels are:")
print(TestLabel)
