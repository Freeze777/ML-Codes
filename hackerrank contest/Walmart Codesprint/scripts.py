
# coding: utf-8

# In[1]:

import pandas as pd
from scipy.sparse import vstack,csr_matrix,bsr_matrix,hstack
import string
import numpy as np
import os
import pickle
import operator
from random import random,shuffle
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer


# In[2]:

df_train=pd.read_csv("dataset/train.tsv",sep="\t")
df_test=pd.read_csv("dataset/test.tsv",sep="\t")


# In[3]:

empty_cols={}
for col in df_train.columns:
    if sum(df_train[col].isnull())>0:
        if col not in empty_cols:
            empty_cols[col]=100*sum(df_train[col].isnull())/float(len(df_train))
#print (empty_cols.items())


# In[4]:

df_train['Item Class ID']=df_train['Item Class ID'].fillna(df_train['Item Class ID'].mode().values[0])
df_test['Item Class ID']=df_test['Item Class ID'].fillna(df_test['Item Class ID'].mode().values[0])


# In[5]:

test_category=(df_test['Item Class ID'].values)
train_category=(df_train['Item Class ID'].values)


# In[6]:

df_category_test=df_test['Item Class ID']


# In[7]:

shelf_tags=set(['1071165','127175','529295','106546','3304195','4538','645319','4536','4537','1084835','1229818','1229819','1229817','5065','95987','1085065','1070524','1229820','447913','650659','522484','648819','1225174','62056','1180168','1229825','1229823','133270','1229821','4483','4457','581514'])


# In[8]:

len(shelf_tags)


# In[9]:

categories=df_train['Item Class ID'].unique()
category_shelf_count={}
for category in categories:
    category_shelf_count[category]={}
    for shelf in shelf_tags:
        category_shelf_count[category][shelf]=0
for i,row in df_train.iterrows():
    temp=(row['tag'].strip('[]').strip().split(','))
    shelves=[x.strip() for x in temp]
    category=row['Item Class ID']
    for shelf in shelves:
        category_shelf_count[category][shelf]+=1


# In[10]:

for category in category_shelf_count:
    category_shelf_count[category]=sorted(category_shelf_count[category].items(), key=operator.itemgetter(1),reverse=True)


# In[11]:

#category_shelf_count


# In[12]:

#category_shelf_count
#total count of '4537' = 2803


# In[13]:

# unfilled columns vs % emptiness
for item in empty_cols.items():
    print item


# In[14]:

columns_to_be_removed=[]
for item in empty_cols.items():
    if item[1]<=65:
        df_train[item[0]]=df_train[item[0]].fillna(df_train[item[0]].mode().values[0])
        df_test[item[0]]=df_test[item[0]].fillna(df_test[item[0]].mode().values[0])
        #columns_to_be_removed.append(item[0])


# In[15]:

print columns_to_be_removed


# In[16]:

df_train=df_train.drop(columns_to_be_removed,axis=1)
df_test=df_test.drop(columns_to_be_removed,axis=1)


# In[17]:

columns_to_processed=set(empty_cols.keys())-set(columns_to_be_removed)
print columns_to_processed


# In[18]:

description_columns=["Product Short Description","Product Long Description","Short Description","Product Name"]
other_columns=["Seller","Actual Color","Item Class ID"]


# In[19]:

#description_columns+=other_columns
description_columns=columns_to_processed


# In[20]:

for col in description_columns:
    df_train[col]=df_train[col].fillna(" ")
    df_test[col]=df_test[col].fillna(" ")
"""for col in other_columns:
    df_train[col]=df_train[col].fillna(df_train[col].mode().values[0])
    df_test[col]=df_test[col].fillna(df_test[col].mode().values[0])"""


# In[21]:

#description_columns+=other_columns


# In[22]:

df_train['Description']=[""]*len(df_train)
df_test['Description']=[""]*len(df_test)
for col in description_columns:
    df_train['Description']+=" "+df_train[col].apply(str)
    df_test['Description']+=" "+df_test[col].apply(str)


# In[23]:

print sum(df_train['Description'].isnull())
print sum(df_test['Description'].isnull())


# In[24]:

import re
import string
def striphtml(data):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    p = re.compile(r'<.*?>')
    #pp= re.compile(r'short description is not available')
    text = p.sub(' ', data).translate(replace_punctuation)
    return text


# In[25]:

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return " ".join(stemmed)

def pre_process(text):
    text=text.decode('utf-8').strip()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


# In[26]:

#print striphtml("i<li>love getting</li>loved and the get loving to seeing the seen")
#print pre_process(striphtml("i<li>love getting</li>loved and the get loving to seeing the seen"))


# In[27]:

df_train["Description"]=df_train['Description'].apply(striphtml)
df_test["Description"]=df_test['Description'].apply(striphtml)


# In[28]:

#df_train["Description"]=df_train['Description'].apply(pre_process)
#df_test["Description"]=df_test['Description'].apply(pre_process)


# In[ ]:




# In[29]:

df_train=df_train.drop(description_columns,axis=1)
df_test=df_test.drop(description_columns,axis=1)


# In[30]:

columns_to_processed=set(columns_to_processed)-set(description_columns)


# In[31]:

print columns_to_processed


# In[32]:

for col in columns_to_processed:
    #print col,df_train[col].mode().values[0]
    df_train[col]=df_train[col].fillna(df_train[col].mode().values[0])
    df_test[col]=df_test[col].fillna(df_test[col].mode().values[0])


# In[33]:

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer


# In[34]:

#vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),sublinear_tf=True)
vectorizer=CountVectorizer(stop_words='english',ngram_range=(1,2))
#vectorizer=HashingVectorizer(stop_words='english',ngram_range=(1,2))


# In[35]:

vectorizer.fit(df_train['Description'].append(df_test['Description']))


# In[36]:

bow_train=vectorizer.transform(df_train['Description'])
bow_test=vectorizer.transform(df_test['Description'])


# In[37]:

"""print bow_train.shape
print bow_test.shape
train_category=np.reshape(train_category,(10593,1))
test_category=np.reshape(test_category,(10593,1))
print test_category.shape
print train_category.shape"""


# In[38]:

"""enc=LabelEncoder()
enc.fit(list(train_category)+list(test_category))
train_category=enc.transform(train_category)
test_category=enc.transform(test_category)"""


# In[39]:

#bow_train=hstack((bow_train,train_category.astype(float)),format='csr')
#bow_test=hstack((bow_test,test_category.astype(float)),format='csr')
#hstack((bow_train, train_category),format='csr')


# In[40]:

labels=[]
for i,row in df_train.iterrows():
    temp=(row['tag'].strip('[]').strip().split(','))
    #print temp
    t=[x.strip() for x in temp]
    #print t
    labels.append(t)
#labels


# In[41]:

binarizer=MultiLabelBinarizer()
bin_multilabels=binarizer.fit_transform(labels)


# In[42]:

print bin_multilabels[0:5]
print labels[0:5]


# In[43]:

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.svm import SVC,LinearSVC
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier,RidgeClassifier,RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model.perceptron import Perceptron
from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_selection import chi2,SelectFromModel,SelectKBest


# In[63]:

"""semi_sup_labels=np.zeros(shape=(len(df_test),32))
semi_sup_labels.fill(-1)
bow_full=vstack((bow_train,bow_test))
labels_full=np.vstack((bin_multilabels,semi_sup_labels))"""


# In[64]:

#model = OneVsRestClassifier(LinearSVC(multi_class='ovr',max_iter=1000,random_state=9701)) #769
#model = OneVsRestClassifier(GaussianNB())
#model = OneVsRestClassifier(LogisticRegression(multi_class='ovr'))
#model = OneVsRestClassifier(RidgeClassifier())
model = OneVsRestClassifier(RidgeClassifierCV())
#model = OneVsRestClassifier(MultinomialNB()) #610
#model = OneVsRestClassifier(AdaBoostClassifier())
#model = OneVsRestClassifier(SGDClassifier(n_iter=100,loss='log')) #689
#model = OneVsRestClassifier(Perceptron(n_iter=1000)) #756
#model = OneVsRestClassifier(RandomForestClassifier(n_estimators=150)) #611
#model= OneVsRestClassifier(XGBClassifier())
model.fit(bow_train,bin_multilabels)


# In[65]:

predictions=model.predict(bow_test)


# In[67]:

"""from scipy.sparse import vstack
bow_full=vstack((bow_train,bow_test))
print type(bin_multilabels),type(predictions)
labels_full=np.vstack((bin_multilabels,predictions))"""


# In[68]:

#semi supervised approach 
"""model = OneVsRestClassifier(LogisticRegression(multi_class='ovr'))
model.fit(bow_full,labels_full)
predictions=model.predict(bow_test)"""


# In[69]:

predictions=binarizer.inverse_transform(predictions)


# In[70]:

#some products are not assigned a shelf
count=0
for t in predictions:
    count+=(len(t)==0)
print count


# In[71]:

shelf_popularity_count={}
for i in range(len(labels)):
    for shelf in labels[i]:
        if shelf not in shelf_popularity_count:
            shelf_popularity_count[shelf]=0
        shelf_popularity_count[shelf]+=1
print shelf_popularity_count


# In[72]:

shelf_popularity_count = sorted(shelf_popularity_count.items(), key=operator.itemgetter(1),reverse=True)


# In[73]:

shelf_popularity_count[0:3]


# In[74]:

#row_num=0
final=[]
for i,category in df_category_test.iteritems():
    t=predictions[i]
    if len(t)==0:
        if category in category_shelf_count:
            #final.append('['+category_shelf_count[category][0][0]+',4537]')
            final.append('['+category_shelf_count[category][0][0]+']')
        else:
            final.append('[4537]')     
    else:
        temp='['+",".join(t)+']'
        final.append(temp)
print final[0:20]
print predictions[0:20]


# In[75]:

df_test['tag']=final


# In[76]:

#df_test.head()


# In[77]:

df_submission=df_test[['item_id','tag']]


# In[78]:

df_submission.to_csv("submission.tsv",sep="\t",index=False)


# In[79]:

#vectorizer.vocabulary_


# In[80]:

#for category in category_shelf_count:
    #print category,category_shelf_count[category][0:3]
#temp=striphtml('<table class="nomobile" style="width: 254px;" border="0" cellspacing="4" align="right"><tbody><tr><td><img title="Surge Suppressor" alt="Surge Suppressor" src="http://images.highspeedbackbone.net/itemdetails/logos/badge-surge-suppressor.gif" /></td><td><img title="12 Outlets" alt="12 Outlets" src="http://images.highspeedbackbone.net/itemdetails/logos/badge-12-outlets.gif" /></td></tr></tbody></table><p><strong>Tripp Lite TLP1208TEL Protect It! Surge Suppressor</strong> <br />Provide high level security for your home and workplace with the Tripp Lite Protect It! Surge Suppressor. The advanced design elements of this 12 Outlet Surge Suppressor gives total protection for your all modern electronic devices from power surges, and spikes. It also provides a right-angle plug, an 8-inch cord, and diagnostic LEDs that helps organize your cables. It also offers powerful surge suppression with 2160 Joules of protection and has 4 transformer-spaced outlet. The RJ11 jacks takes supreme control of your your modem/fax/phone equipments. Ensure the protection of all connected electronics using the Tripp Lite Protect It! Surge Suppressor.</p><p><strong>What It Is and Why You Need It:</strong></p><ul><li>12 Outlet; gives total protection for your all modern electronic devices</li><li>8-inch cord; helps organize your cables</li><li>2160 joules; protects equipment from line noise and the strongest surges</li><li>RJ11 jacks; secures your modem/fax/phone equipment from damages</li></ul><center class="nomobile"><p><img src="http://images.highspeedbackbone.net/SKUimages/enhanced/TLP1208TEL.jpg"></p></center>')


# In[81]:

#print vectorizer.transform(temp)

