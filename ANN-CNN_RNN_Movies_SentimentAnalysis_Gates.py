# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:50:30 2022

@author: profa
"""
import numpy as np
import nltk
import pandas as pd
import sklearn
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import tensorflow

##################################################################################
# ANN - CNN - RNN
# Movies Dataset
# https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
##################################################################################
TrainData = pd.read_csv("Train.csv")
print(TrainData.head())

TestData = pd.read_csv("Test.csv")

ValidData = pd.read_csv("Valid.csv")

# Clean Up Data, Tokenize and Vectorize

ReviewsLIST = []  ## from the text column
LabelLIST = []

for nextreview, nextlabel in zip(TrainData["text"], TrainData["label"]):
    ReviewsLIST.append(nextreview)
    LabelLIST.append(nextlabel)

# ----------------------------------------
# Use NLTK's PorterStemmer in a function - Stemming
# -------------------------------------------------------
A_STEMMER = PorterStemmer()


def MY_STEMMER(str_input):
    # Only use letters, no punct, no nums, make lowercase...
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [A_STEMMER.stem(word) for word in words]
    return words


#########################################
#  Build the labeled dataframe
#  Get the Vocab  - here keeping top 10,000
######################################################

# Vectorize
# Instantiate your CV
MyCountV = CountVectorizer(
    input="content",
    lowercase=True,
    stop_words="english",  # Remove stopwords
    tokenizer=MY_STEMMER,  # Stemming
    max_features=10000  # This can be updated
)

# Use your CV
MyDTM = MyCountV.fit_transform(ReviewsLIST)  # create a sparse matrix
ColumnNames = MyCountV.get_feature_names()  # This is the vocab

# clean up the columns
MyDTM_DF = pd.DataFrame(MyDTM.toarray(), columns=ColumnNames)
# Convert the labels from list to df
Labels_DF = pd.DataFrame(LabelLIST, columns=['LABEL'])

for nextcol in MyDTM_DF.columns:
    # The following will remove all columns that contains numbers
    if str.isdigit(nextcol):
        MyDTM_DF = MyDTM_DF.drop([nextcol], axis=1)

    # The following will remove any column with name of 3 or smaller - like "it" or "of" or "pre".
    elif len(str(nextcol)) < 3:
        print(nextcol)
        MyDTM_DF = MyDTM_DF.drop([nextcol], axis=1)

# Save original DF - without the lables
My_Orig_DF = MyDTM_DF

## Now - let's create a complete and labeled
## dataframe:
dfs = [Labels_DF, MyDTM_DF]
print(dfs)

Final_DF_Labeled = pd.concat(dfs, axis=1, join='inner')
## DF with labels
print(Final_DF_Labeled.iloc[:, 0:2])
print(Final_DF_Labeled.shape)

## Create list of all words
print(Final_DF_Labeled.columns[0])
NumCols = Final_DF_Labeled.shape[1]
print(NumCols)
print(len(list(Final_DF_Labeled.columns)))

top_words = list(Final_DF_Labeled.columns[1:NumCols + 1])
## Exclude the Label

print(top_words[0])
print(top_words[-1])

print(type(top_words))
print(top_words.index("aamir"))  ## index 0 in top_words
print(top_words.index("zucco"))  # index NumCols - 2 in top_words


## Encoding the data
def Encode(review):
    words = review.split()
    # print(words)
    if len(words) > 500:
        words = words[:500]
        # print(words)
    encoding = []
    for word in words:
        try:
            index = top_words.index(word)
        except:
            index = (NumCols - 1)
        encoding.append(index)
    while len(encoding) < 500:
        encoding.append(NumCols)
    return encoding


##-------------------------------------------------------
## Test the code to assure that it is
## doing what you think it should

result1 = Encode("aaron aamir abbey abbott abilities zucco ")
print(result1)
result2 = Encode("york young younger youngest youngsters youth youthful youtube zach zane zany zealand zellweger")
print(result2)
print(len(result2))  ## Will be 500 because we set it that way above
##-----------------------------------------------------------

###################################
## Now we are ready to encode all of our
## reviews - which are called "text" in
## our dataset.

# Using vocab from above i -  convert reviews (text) into numerical form
# Replacing each word with its corresponding integer index value from the
# vocabulary. Words not in the vocab will
# be assigned  as the max length of the vocab + 1
## ########################################################

# Encode our training and testing datasets
# with same vocab.

print(TestData.head(10))
print(TestData.shape)
print(TrainData.shape)

############### Final Training and Testing data and labels-----------------
training_data = np.array([Encode(review) for review in TrainData["text"]])
print(training_data[20])
print(training_data.shape)

testing_data = np.array([Encode(review) for review in TestData['text']])
print(testing_data[20])

validation_data = np.array([Encode(review) for review in ValidData['text']])

print(training_data.shape, testing_data.shape)

## Prepare the labels if they are not already 0 and 1. In our case they are
## so these lines are commented out and just FYI
# train_labels = [1 if label=='positive' else 0 for sentiment in TrainData['label']]
# test_labels = [1 if label=='positive' else 0 for sentiment in TestData['label']]
train_labels = np.array([TrainData['label']])
train_labels = train_labels.T
print(train_labels.shape)
test_labels = np.array([TestData['label']])
test_labels = test_labels.T
print(test_labels.shape)
