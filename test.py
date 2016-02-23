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

X_train, X_test, Y_train, Y_test = train_test_split(Summary, Score, test_size=0.2, random_state=42)

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
for text in X_test:
    text = text.lower()
    text = remove_symbols(text)
    text=tokenize(text)
    test_set.append(text)

X_new_counts = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

##from pandas import *
##df = DataFrame({'Before': X_train, 'After': corpus})
##print(df.head(20))

prediction = dict()

## NB

##TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
##from sklearn.naive_bayes import GaussianNB
##model = GaussianNB().fit(X_train_tfidf, Y_train)
##prediction['Naive'] = clf.predict(X_test_tfidf)
##print("Naive Bayes")
##print(metrics.classification_report(Y_test, prediction['Naive']))

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train_tfidf, Y_train)
prediction['Multinomial'] = model.predict(X_test_tfidf)
print("Multinomial")
print(metrics.classification_report(Y_test, prediction['Multinomial']))

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X_train_tfidf, Y_train)
prediction['Bernoulli'] = model.predict(X_test_tfidf)
print("Bernoulli")
print(metrics.classification_report(Y_test, prediction['Bernoulli']))


## SVM

##UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
##from sklearn.svm import SVC
##model = SVC().fit(X_train_tfidf, Y_train)
##prediction['SVC'] = model.predict(X_test_tfidf)
##print("SVC")
##print(metrics.classification_report(Y_test, prediction['SVC']))

from sklearn.svm import LinearSVC
model = LinearSVC().fit(X_train_tfidf, Y_train)
prediction['LinearSVC'] = model.predict(X_test_tfidf)
print("LinearSVC")
print(metrics.classification_report(Y_test, prediction['LinearSVC']))

##ValueError: specified nu is infeasible
##from sklearn.svm import NuSVC
##model = NuSVC().fit(X_train_tfidf, Y_train)
##prediction['SVC'] = model.predict(X_test_tfidf)
##print("NuSVC")
##print(metrics.classification_report(Y_test, prediction['NuSVC']))

## Regression
from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5, multi_class='ovr')
logreg.fit(X_train_tfidf, Y_train)
prediction['Logistic'] = logreg.predict(X_test_tfidf)
print("Logistic")
print(metrics.classification_report(Y_test, prediction['Logistic']))







