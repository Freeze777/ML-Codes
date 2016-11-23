
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import pickle
import os
import re
import string
from random import shuffle
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC,LinearSVC
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier,RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model.perceptron import Perceptron
from sklearn.cross_validation import train_test_split
from lda import LDA
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold,cross_val_score


# In[2]:

arf_path = 'ARF_beforeProcessing/'
non_arf_path = 'NonARF_beforeProcessing/'


# In[3]:

def strip_punctuations(data):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = data.translate(replace_punctuation)
    text = re.sub(' +',' ',text)
    return text


# In[4]:

stemmer = PorterStemmer()
def stem_text(data):
    #return data
    tokens = nltk.word_tokenize(data)
    stemmed = []
    for item in tokens:
        if not item.isdigit():#to remove word which is a number
            stemmed.append(stemmer.stem(item))
    return " ".join(stemmed)


# In[5]:

s="982476,.'\][;]...,,,||}{)(!@#@$%^&*fre#@1ez!!e love loving loved doing done do"
print stem_text(strip_punctuations(s))


# In[6]:

#note=open(arf_path+'26446.0.0_15399.0.0.txt', 'r')
#note=stem_text(strip_punctuations(note.read()))
#print note


# In[7]:

arf_notes=[]
arf_labels=[]

for filename in os.listdir(arf_path):
    note=open(arf_path+filename, 'r')
    note=stem_text(strip_punctuations(note.read()))
    arf_notes.append(note)
    arf_labels.append(-1)


# In[8]:

non_arf_notes=[]
non_arf_labels=[]

for filename in os.listdir(non_arf_path):
    note=open(non_arf_path+filename, 'r')
    note=stem_text(strip_punctuations(note.read()))
    non_arf_notes.append(note)
    non_arf_labels.append(1)


# In[9]:

corpus=arf_notes+non_arf_notes
labels=arf_labels+non_arf_labels


# In[10]:

tf_idf_vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,1))#,sublinear_tf=True)
tf_idf_vectorizer.fit(corpus)
#top meaningful words
indices = np.argsort(tf_idf_vectorizer.idf_)#[::-1]
features = tf_idf_vectorizer.get_feature_names()
top_n = int(0.95*(len(tf_idf_vectorizer.vocabulary_)))
low_n= len(tf_idf_vectorizer.vocabulary_)-top_n
vocabulary = [features[i] for i in indices[low_n//2:(len(tf_idf_vectorizer.vocabulary_)-(low_n//2))]]#indices[0:top_n]]
#print vocabulary
print len(tf_idf_vectorizer.vocabulary_)
print len(vocabulary)


# In[11]:

vectorizer=CountVectorizer(stop_words='english',vocabulary=vocabulary)#,ngram_range=(1,1))
#bag_of_words=vectorizer.fit_transform(corpus)
bag_of_words=vectorizer.transform(corpus)


# In[12]:

len(vectorizer.vocabulary_)


# In[13]:

print len(corpus)


# In[14]:

# suspicious empty file detected!! '11464.0.0_11825.0.0.txt'
#for i in range(len(corpus)):
#    print bag_of_words[i].sum()


# In[15]:

#8,10,5
num_topics=9


# In[ ]:

#scikit-learn LDA implementation
#201
#1121
#4617
#model=LatentDirichletAllocation(n_topics=num_topics,max_iter=100,learning_method='batch',random_state=201)#,doc_topic_prior=50.0/num_topics,topic_word_prior=200.0/num_topics)
#model.fit(bag_of_words)


# In[ ]:




# In[ ]:

#lda implementation from https://github.com/ariddell/lda using collapsed gibbs sampling
model = LDA(n_topics=num_topics, n_iter=1000, random_state=201,refresh=100)
model.fit(bag_of_words)  # model.fit_transform(X) is also available
#topic_word = model.topic_word_  # model.components_ also works


# In[ ]:

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# In[ ]:

feature_names = vectorizer.get_feature_names()
print_top_words(model, feature_names, 10)


# In[ ]:

lda_out=model.transform(bag_of_words)


# In[ ]:

#probalitize each documents topic distribution
for i in range(len(lda_out)):
    lda_out[i]=(lda_out[i]/float(sum(lda_out[i])))


# In[ ]:

df = pd.DataFrame(lda_out)
df['label']=labels
df_shuff=df.sample(frac=1,random_state=421).reset_index(drop=True)


# In[ ]:

features=range(num_topics)


# In[ ]:

x_train, x_test, y_train, y_test = train_test_split(df_shuff[features],df_shuff['label'], test_size=1/4.0, random_state=4082,stratify=df_shuff['label'])


# In[ ]:

print "train","#label(1):",sum(y_train==1)," #label(-1):",sum(y_train==-1)
print "test","#label(1):",sum(y_test==1)," #label(-1):",sum(y_test==-1)


# In[ ]:

#clf= XGBClassifier()
#clf = AdaBoostClassifier()
#clf = RandomForestClassifier(n_estimators=150)
#clf = RidgeClassifier()
#clf = LinearSVC(max_iter=100)#,random_state=9701)
clf=SVC(C=100.0,coef0=0.0,degree=3, gamma='auto', kernel='linear',max_iter=100,shrinking=True,tol=0.001)


# In[ ]:

clf.fit(x_train,y_train)
test_pred=clf.predict(x_test)
train_pred=clf.predict(x_train)


# In[ ]:

print "#miss-classification (train): ",sum(train_pred!=y_train),"/",len(train_pred)
print "#miss-classification (test): ",sum(test_pred!=y_test),"/",len(test_pred)
print "train accuracy: ",100.0*sum(train_pred==y_train)/float(len(train_pred)),"%"
print "test accuracy: ",100.0*sum(test_pred==y_test)/float(len(test_pred)),"%"


# In[ ]:

clf=SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, 
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
   max_iter=100, probability=False, random_state=None, shrinking=True,
   tol=0.001, verbose=False)

# Cross Validation
scores=cross_val_score(clf, df_shuff[features], df_shuff['label'], cv = 5)

# Prining the results
print('Prediction result')
print(scores)
print("Accuracy: %0.2f(+/- %0.2f)" %(scores.mean(), scores.std()*2))


# In[ ]:

"""vectorizer=TfidfVectorizer(stop_words='english')#,sublinear_tf=True)
vectorizer.fit(arf_notes+non_arf_notes)
#top meaningful words
indices = np.argsort(vectorizer.idf_)[::-1]
features = vectorizer.get_feature_names()
top_n = 50
top_features = [features[i] for i in indices[:top_n]]
print top_features
print len(vectorizer.vocabulary_)
#print vectorizer.vocabulary_['5020']"""

