import re
import csv
import sqlite3
import mysql.connector
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import pickle

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


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


counts = open("/Users/linlu/Desktop/sentiment/counts.pickle", "rb")
count_vect = pickle.load(counts)
counts.close()

tfidf = open("/Users/linlu/Desktop/sentiment/tfidf.pickle", "rb")
tfidf_transformer = pickle.load(tfidf)
tfidf.close()

open_file = open("/Users/linlu/Desktop/sentiment/clf1.pickle","rb")
clf1 = pickle.load(open_file)
open_file.close()


open_file = open("/Users/linlu/Desktop/sentiment/clf2.pickle","rb")
clf2 = pickle.load(open_file)
open_file.close()


open_file = open("/Users/linlu/Desktop/sentiment/clf3.pickle","rb")
clf3 = pickle.load(open_file)
open_file.close()


open_file = open("/Users/linlu/Desktop/sentiment/vclf.pickle","rb")
vclf = pickle.load(open_file)
open_file.close()
#--- Test set

cnx = mysql.connector.connect(user='lzl0032', password = 'bCM2g7KQ5c',
                              host='131.204.67.247',
                              database='peerjust')

cursor = cnx.cursor()

testmessages = pd.read_sql_query("""
SELECT comment_id, message
FROM comment_info
WHERE comment_id BETWEEN 175501 AND 178877
""", cnx)

index = testmessages['comment_id']
X_test = testmessages['message']

test_set = []

for text in X_test:
    text = text.lower()
    text = remove_symbols(text)
    text = tokenize(text)
    test_set.append(text)
          
X_new_counts = count_vect.transform(test_set)

X_test_tfidf = tfidf_transformer.transform(X_new_counts)

prediction = dict()

prediction['LinearSVC'] = clf1.predict(X_test_tfidf)

prediction['Logreg'] = clf2.predict(X_test_tfidf)

prediction['PassAgg'] = clf3.predict(X_test_tfidf)

prediction['Vote'] = vclf.predict(X_test_tfidf)

## Output the classification results into mysql

col1 = prediction['LinearSVC']
col2 = prediction['Logreg']
col3 = prediction['PassAgg']
col4 = prediction['Vote']

j = 0

for i in index:
    cursor.execute("UPDATE comment_info SET linearsvc = %s, logreg = %s, passagg = %s, vote = %s WHERE comment_id = %s ",
                   (str(col1[j]),str(col2[j]),str(col3[j]),str(col4[j]),int(i)))
    j = j + 1
        
cnx.commit()

cursor.close()
cnx.close()
