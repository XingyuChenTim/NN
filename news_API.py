# Data gathering via API - URLs and GET
# Cleaning and preparing text DATA
# DTM and Data Frames
# Training and Testing at DT

#########################################
# https://newsapi.org/ for register an API key
#########################################

# Example URL
# https://newsapi.org/v2/everything?
# q=tesla&from=2021-05-20&sortBy=publishedAt&
# apiKey="1df08f5f5fd54b8eb321e7d512acd363"

# What to import
import requests
import re
import pandas as pd
from pandas import DataFrame

## To tokenize and vectorize text type data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# import PIL
# import Pillow
# import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
# from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import graphviz

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn
from sklearn.cluster import KMeans

from sklearn import preprocessing

import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram

####################################
##
##  Step 1: Connect to the server
##          Send a query
##          Collect and clean the
##          results
####################################

####################################################
##In the following loop, we will query the newsapi servers
##for all the topic names in the list
## We will then build a large csv file
## where each article is a row
##
## From there, we will convert this data
## into a labeled dataframe
## so we can train and then test our DT
## model
####################################################

####################################################
## Build the URL and GET the results
## NOTE: At the bottom of this code
## commented out, you will find a second
## method for doing the following. This is FYI.
####################################################

## This is the endpoint - the server and
## location on the server where your data
## will be retrieved from

## TEST FIRST!
## We are about to build this URL:
## https://newsapi.org/v2/everything?apiKey=8f4134 your key here 0f22b&q=bitcoin


topics = ["abortion", "election"]

## topics needs to be a list of strings (words)
## Next, let's build the csv file
## first and add the column names
## Create a new csv file to save the headlines
filename = "NewHeadlines.csv"
MyFILE = open(filename, "w")  # "a"  for append   "r" for read
## with open
### Place the column names in - write to the first row
WriteThis = "LABEL,Date,Source,Title,Headline\n"
MyFILE.write(WriteThis)
MyFILE.close()

## CHeck it! Can you find this file?

#### --------------------> GATHER - CLEAN - CREATE FILE

## RE: documentation and options
## https://newsapi.org/docs/endpoints/everything

endpoint = "https://newsapi.org/v2/everything"

################# enter for loop to collect
################# data on three topics
#######################################

for topic in topics:

    ## Dictionary Structure
    URLPost = {'apiKey': '1df08f5f5fd54b8eb321e7d512acd363',
               'q': topic
               }

    response = requests.get(endpoint, URLPost)
    print(response)
    jsontxt = response.json()
    print(jsontxt)
    #####################################################

    ## Open the file for append
    MyFILE = open(filename, "a")
    LABEL = topic
    for items in jsontxt["articles"]:
        print(items, "\n\n\n")

        # Author=items["author"]
        # Author=str(Author)
        # Author=Author.replace(',', '')

        Source = items["source"]["name"]
        print(Source)

        Date = items["publishedAt"]
        ##clean up the date
        NewDate = Date.split("T")
        Date = NewDate[0]
        print(Date)

        ## CLEAN the Title
        ##----------------------------------------------------------
        ##Replace punctuation with space
        # Accept one or more copies of punctuation
        # plus zero or more copies of a space
        # and replace it with a single space
        Title = items["title"]
        Title = str(Title)
        # print(Title)
        Title = re.sub(r'[,.;@#?!&$\-\']+', ' ', str(Title), flags=re.IGNORECASE)
        Title = re.sub(' +', ' ', str(Title), flags=re.IGNORECASE)
        Title = re.sub(r'\"', ' ', str(Title), flags=re.IGNORECASE)

        # and replace it with a single space
        ## NOTE: Using the "^" on the inside of the [] means
        ## we want to look for any chars NOT a-z or A-Z and replace
        ## them with blank. This removes chars that should not be there.
        Title = re.sub(r'[^a-zA-Z]', " ", str(Title), flags=re.VERBOSE)
        Title = Title.replace(',', '')
        Title = ' '.join(Title.split())
        Title = re.sub("\n|\r", "", Title)
        print(Title)
        ##----------------------------------------------------------

        Headline = items["description"]
        Headline = str(Headline)
        Headline = re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
        Headline = re.sub(' +', ' ', Headline, flags=re.IGNORECASE)
        Headline = re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
        Headline = re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)
        ## Be sure there are no commas in the headlines or it will
        ## write poorly to a csv file....
        Headline = Headline.replace(',', '')
        Headline = ' '.join(Headline.split())
        Headline = re.sub("\n|\r", "", Headline)

        ### AS AN OPTION - remove words of a given length............
        Headline = ' '.join([wd for wd in Headline.split() if len(wd) > 3])

        # print("Author: ", Author, "\n")
        # print("Title: ", Title, "\n")
        # print("Headline News Item: ", Headline, "\n\n")

        # print(Author)
        print(Title)
        print(Headline)

        WriteThis = str(LABEL) + "," + str(Date) + "," + str(Source) + "," + str(Title) + "," + str(Headline) + "\n"
        print(WriteThis)

        MyFILE.write(WriteThis)

    ## CLOSE THE FILE
    MyFILE.close()

################## END for loop

####################################################
##
## Where are we now?
##
## So far, we have created a csv file
## with labeled data. Each row is a news article
##
## - BUT -
## We are not done. We need to choose which
## parts of this data to use to model our decision tree
## and we need to convert the data into a data frame.
##
########################################################


BBC_DF = pd.read_csv(filename, error_bad_lines=False)
print(BBC_DF.head())
# iterating the columns
for col in BBC_DF.columns:
    print(col)

print(BBC_DF["Headline"])

## REMOVE any rows with NaN in them
BBC_DF = BBC_DF.dropna()
print(BBC_DF["Headline"])

### Tokenize and Vectorize the Headlines
## Create the list of headlines
## Keep the labels!

HeadlineLIST = []
LabelLIST = []

for nexthead, nextlabel in zip(BBC_DF["Headline"], BBC_DF["LABEL"]):
    HeadlineLIST.append(nexthead)
    LabelLIST.append(nextlabel)

print("The headline list is:\n")
print(HeadlineLIST)

print("The label list is:\n")
print(LabelLIST)

##########################################
## Remove all words that match the topics.
## For example, if the topics are food and covid
## remove these exact words.
##
## We will need to do this by hand.
NewHeadlineLIST = []

for element in HeadlineLIST:
    print(element)
    print(type(element))
    ## make into list
    AllWords = element.split(" ")
    print(AllWords)

    ## Now remove words that are in your topics
    NewWordsList = []
    for word in AllWords:
        print(word)
        word = word.lower()
        if word in topics:
            print(word)
        else:
            NewWordsList.append(word)

    ##turn back to string
    NewWords = " ".join(NewWordsList)
    ## Place into NewHeadlineLIST
    NewHeadlineLIST.append(NewWords)

##
## Set the     HeadlineLIST to the new one
HeadlineLIST = NewHeadlineLIST
print(HeadlineLIST)
#########################################
##
##  Build the labeled dataframe
##
######################################################

### Vectorize
## Instantiate your CV
MyCountV = CountVectorizer(
    input="content",  ## because we have a csv file
    lowercase=True,
    stop_words="english",
    max_features=50
)

## Use your CV
MyDTM = MyCountV.fit_transform(HeadlineLIST)  # create a sparse matrix
print(type(MyDTM))

ColumnNames = MyCountV.get_feature_names()
# print(type(ColumnNames))


## Build the data frame
MyDTM_DF = pd.DataFrame(MyDTM.toarray(), columns=ColumnNames)

## Convert the labels from list to df
Labels_DF = DataFrame(LabelLIST, columns=['LABEL'])

## Check your new DF and you new Labels df:
print("Labels\n")
print(Labels_DF)
print("News df\n")
print(MyDTM_DF.iloc[:, 0:6])

##Save original DF - without the lables
My_Orig_DF = MyDTM_DF
print(My_Orig_DF)
######################
## AND - just to make sure our dataframe is fair
## let's remove columns called:
## food, bitcoin, and sports (as these are label names)
######################
# MyDTM_DF=MyDTM_DF.drop(topics, axis=1)


## Now - let's create a complete and labeled
## dataframe:
dfs = [Labels_DF, MyDTM_DF]
print(dfs)

Final_News_DF_Labeled = pd.concat(dfs, axis=1, join='inner')
## DF with labels
print(Final_News_DF_Labeled)

#############################################
##
## Create Training and Testing Data
##
## Then model and test the Decision Tree
##
################################################


## Before we start our modeling, let's visualize and
## explore.

##It might be very interesting to see the word clouds
## for each  of the topics.
##--------------------------------------------------------
List_of_WC = []

for mytopic in topics:
    tempdf = Final_News_DF_Labeled[Final_News_DF_Labeled['LABEL'] == mytopic]
    print(tempdf)

    tempdf = tempdf.sum(axis=0, numeric_only=True)
    # print(tempdf)

    # Make var name
    NextVarName = str("wc" + str(mytopic))
    # print( NextVarName)

    ##In the same folder as this code, I have three images
    ## They are called: food.jpg, bitcoin.jpg, and sports.jpg
    # next_image=str(str(mytopic) + ".jpg")
    # print(next_image)

    ## https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html

    ###########
    ## Create and store in a list the wordcloud OBJECTS
    #########
    NextVarName = WordCloud(width=1000, height=600, background_color="white",
                            min_word_length=4,  # mask=next_image,
                            max_words=400).generate_from_frequencies(tempdf)

    ## Here, this list holds all three wordclouds I am building
    List_of_WC.append(NextVarName)

##------------------------------------------------------------------
print(List_of_WC)
##########
########## Create the wordclouds
##########
fig = plt.figure(figsize=(25, 25))
# figure, axes = plt.subplots(nrows=2, ncols=2)
NumTopics = len(topics)
for i in range(NumTopics):
    print(i)
    ax = fig.add_subplot(NumTopics, 1, i + 1)
    plt.imshow(List_of_WC[i], interpolation='bilinear')
    plt.axis("off")
    plt.savefig("NewClouds.pdf")
