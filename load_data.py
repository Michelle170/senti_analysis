import re
import csv
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

con = sqlite3.connect('/Users/linlu/Desktop/aff/database.sqlite')

messages = pd.read_sql_query("""
SELECT Score, Summary
FROM Reviews
""", con)

def partition(x):
    if x < 3:
        return 'negative'
    elif x == 3:
        return 'neutral'
    return 'positive'

Score = messages['Score']
Score = Score.map(partition)
Summary = messages['Summary']

tmp = messages
tmp['Score'] = tmp['Score'].map(partition)
##
##f = open('training.csv','w')
##print >> f, tmp
##f.close()

X_train = Summary
Y_train = Score

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stems = stem_tokens(tokens, stemmer)
    return ' '.join(stems)

def remove_symbols(text):
    # remove user mentions, tags, urls
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
    return text

#--- Training set

corpus = []
for text in X_train:
    text = text.lower()
    text = remove_symbols(text)
    text=tokenize(text)
    corpus.append(text)
        
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)        
       
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#--- Test set

test_set = []
#X_test1 = []
with open('fbeat2.csv') as f:
    X_test = csv.reader(f)
    for text in X_test:
 #      X_test1.append(text)
        text = ' '.join(text)
        text = text.lower()
        text = remove_symbols(text)
        text=tokenize(text)
        test_set.append(text)

f.close()
        
X_new_counts = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

prediction = dict()

from sklearn.svm import LinearSVC
model = LinearSVC().fit(X_train_tfidf, Y_train)
prediction['LinearSVC'] = model.predict(X_test_tfidf)

f1 = open('fbeat_output.csv','w')
for value in prediction['LinearSVC']:
    print >> f1, value
f1.close()
