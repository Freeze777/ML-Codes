import json
import pandas
import string
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import MultinomialNB
data=[]
f=open("training.json",'r')
n=int(f.readline())
for i in range(n):
       t=json.loads(f.readline())
       t[u'question']=t[u'question']+' '+t[u'excerpt']
       t[u'topic']=str(t[u'topic'])
       t.pop(u'excerpt')
       data.append(t)

data = pandas.DataFrame(data)
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b',stop_words='english')
feature_matrix=vectorizer.fit_transform(data[u'question'])
log_reg=LogisticRegression(multi_class='ovr')
#log_reg=MultinomialNB()
log_reg.fit(feature_matrix,data[u'topic'])


test_data=[]
n=int(raw_input())
for i in range(n):
       t=json.loads(raw_input())
       t[u'question']=t[u'question']+' '+t[u'excerpt']
       t[u'topic']=str(t[u'topic'])
       t.pop(u'excerpt')
       test_data.append(t)

test_data = pandas.DataFrame(test_data)
test_feature_matrix=vectorizer.transform(test_data[u'question'])
pred=log_reg.predict(test_feature_matrix)

for i in range(len(pred)):
	print pred[i]
