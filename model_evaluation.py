import re
import csv
import sqlite3
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

con = sqlite3.connect('/Users/linlu/Desktop/sentiment/aff/database.sqlite')

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

X_train = Summary
Y_train = Score
##X_train, X_test, Y_train, Y_test = train_test_split(Summary, Score, test_size=0.4, random_state=42)

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

save_counts = open("/Users/linlu/Desktop/sentiment/counts.pickle","wb")
pickle.dump(count_vect, save_counts)
save_counts.close()
      
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

save_tfidf = open("/Users/linlu/Desktop/sentiment/tfidf.pickle","wb")
pickle.dump(tfidf_transformer, save_tfidf)
save_tfidf.close()

#--- Test set

##test_set = []
##for text in X_test:
##    text = text.lower()
##    text = remove_symbols(text)
##    text=tokenize(text)
##    test_set.append(text)
##
##X_new_counts = count_vect.transform(test_set)
##X_test_tfidf = tfidf_transformer.transform(X_new_counts)

prediction = dict()

from sklearn import svm
clf1 = svm.LinearSVC(penalty = 'l2',dual=False, class_weight='balanced')

from sklearn import linear_model
clf2 = linear_model.LogisticRegression(multi_class='ovr', class_weight='balanced')

clf3 = linear_model.PassiveAggressiveClassifier(class_weight='balanced')

from sklearn import ensemble
vclf = ensemble.VotingClassifier(estimators=[('LinearSVC', clf1),
                                              ('logreg', clf2),
                                              ('PassAgg', clf3),
                                             ])


clf1.fit(X_train_tfidf, Y_train)
##prediction['LinearSVC'] = clf1.predict(X_test_tfidf)
save_clf = open("/Users/linlu/Desktop/sentiment/clf1.pickle","wb")
pickle.dump(clf1, save_clf)
save_clf.close()


clf2.fit(X_train_tfidf, Y_train)
##prediction['logreg'] = clf2.predict(X_test_tfidf)
save_clf = open("/Users/linlu/Desktop/sentiment/clf2.pickle","wb")
pickle.dump(clf2, save_clf)
save_clf.close()

clf3.fit(X_train_tfidf, Y_train)
##prediction['PassAgg'] = clf3.predict(X_test_tfidf)
save_clf = open("/Users/linlu/Desktop/sentiment/clf3.pickle","wb")
pickle.dump(clf3, save_clf)
save_clf.close()

vclf.fit(X_train_tfidf, Y_train)
##prediction['Vote'] = vclf.predict(X_test_tfidf)
save_clf = open("/Users/linlu/Desktop/sentiment/vclf.pickle","wb")
pickle.dump(vclf, save_clf)
save_clf.close()



##evaluate model using ROC

##def leave_binary(x):
##    x = filter(lambda x: x != 'neutral', x)
##    return x
##
##def formatt(x):
##    leave_binary(x)
##    if x == 'negative':
##        return 0
##    return 1
##vfunc = np.vectorize(formatt)
##
##cmp = 0
##colors = ['b', 'g', 'y', 'c', 'm', 'k']
##for model, predicted in prediction.items():
##    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test.map(formatt), vfunc(predicted))
##    roc_auc = auc(false_positive_rate, true_positive_rate)
##    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
##    cmp += 1
##
##plt.title('Classifiers comparaison with ROC')
##plt.legend(loc='lower right')
##plt.plot([0,1],[0,1],'r--')
##plt.xlim([-0.1,1.2])
##plt.ylim([-0.1,1.2])
##plt.ylabel('True Positive Rate')
##plt.xlabel('False Positive Rate')
##plt.show()
##
##
#### evaluate model using confusion matrix
##
##def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
##    plt.imshow(cm, interpolation='nearest', cmap=cmap)
##    plt.title(title)
##    plt.colorbar()
##    tick_marks = np.arange(len(set(Score)))
##    plt.xticks(tick_marks, set(Score), rotation=45)
##    plt.yticks(tick_marks, set(Score))
##    plt.tight_layout()
##    plt.ylabel('True label')
##    plt.xlabel('Predicted label')
##    
### Compute confusion matrix
##cm = confusion_matrix(Y_test, prediction['Vote'])
##np.set_printoptions(precision=2)
##print('Confusion matrix, without normalization')
##print(cm)
##plt.figure()
##plot_confusion_matrix(cm)    
##
##cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
##print('Normalized confusion matrix')
##print(cm_normalized)
##plt.figure()
##plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
##
##plt.show()
