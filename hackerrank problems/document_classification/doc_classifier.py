import pandas
import string
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import MultinomialNB
label=[]
corpus=[]
f=open("trainingdata.txt",'r')
n=int(f.readline())
for i in range(n):
       t=f.readline()
       l=int(t[0])
       corpus.append(t[2:])
       label.append(l)

data = pandas.DataFrame({'corpus':corpus,'label':label})
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b',stop_words='english')
feature_matrix=vectorizer.fit_transform(data['corpus'])
log_reg=LogisticRegression(multi_class='ovr')
#log_reg=MultinomialNB()
log_reg.fit(feature_matrix,data['label'])


test_corpus=[]
n=int(raw_input())
for i in range(n):
       test_corpus.append(raw_input())

test_data = pandas.DataFrame({'corpus':test_corpus})
test_feature_matrix=vectorizer.transform(test_data['corpus'])
pred=log_reg.predict(test_feature_matrix)

for i in range(len(pred)):
	print pred[i]
