
# coding: utf-8

# In[1]:

import pandas as pd
import string
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder


# In[ ]:

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    return

def load_object(filename):
    with open(filename, 'rb') as input:
        obj=pickle.load(input)
    return obj


# In[2]:

df=pd.read_csv("training_dataset.csv")
df_test=pd.read_csv('test_dataset.csv')
df=df.reindex(np.random.permutation(df.index))

"""user_unsub={}
for i,row in df.iterrows():
	if row['user_id'] not in user_unsub:
		user_unsub[row['user_id']]=set()
	if row['unsubscribed']==True:
		user_unsub[row['user_id']].add(row['mail_category'])

save_object(user_unsub,'user_unsub.pkl')

user_unsub=load_object("user_unsub.pkl")"""

# In[3]:

print len(df.loc[df['opened']==True])
print len(df.loc[df['opened']==False])
print len(df)


# In[4]:

df_opened=df.loc[df['opened']==True]
df_closed=df.loc[df['opened']==False]
percentage = len(df_opened)/float(len(df_closed))

df_closed=df_closed.sample(frac=percentage)
df=df_opened.append(df_closed)
df=df.reindex(np.random.permutation(df.index))


# In[5]:

print len(df.loc[df['opened']==True])
print len(df.loc[df['opened']==False])
print len(df)


# In[ ]:

#df=load_object("df.pkl")


# In[6]:

#features=['mail_category','mail_type']
features=['mail_category','mail_type','sent_time','last_online','hacker_created_at','hacker_timezone','contest_login_count','contest_login_count_1_days','contest_login_count_30_days','contest_login_count_365_days','contest_login_count_7_days','contest_participation_count','contest_participation_count_1_days','contest_participation_count_30_days','contest_participation_count_365_days','contest_participation_count_7_days','forum_comments_count','forum_count','forum_expert_count','forum_questions_count','hacker_confirmation','ipn_count','ipn_count_1_days','ipn_count_30_days','ipn_count_365_days','ipn_count_7_days','ipn_read','ipn_read_1_days','ipn_read_30_days','ipn_read_365_days','ipn_read_7_days','submissions_count','submissions_count_1_days','submissions_count_30_days','submissions_count_365_days','submissions_count_7_days','submissions_count_contest','submissions_count_contest_1_days','submissions_count_contest_30_days','submissions_count_contest_365_days','submissions_count_contest_7_days','submissions_count_master','submissions_count_master_1_days','submissions_count_master_30_days','submissions_count_master_365_days','submissions_count_master_7_days']
dtrain={}
dtest={}
for col in features:
    dtrain[col]=df[col].value_counts().index[0]
    dtest[col]=df_test[col].value_counts().index[0]
    #dtrain[col]=df[col].value_counts().index[0]
    #dtest[col]=df_test[col].value_counts().index[0]
    #df_test = df_test.fillna(df_test[col].value_counts().index[0])
    #df[col]=df[col].fillna(df[col].median())
    #df_test[col]=df_test[col].fillna(df_test[col].median())
    #print col,len(df[df[col].isnull()]),len(df_test[df_test[col].isnull()])
    
for col in features:
    df[col]=df[col].fillna(method='bfill')
    df_test[col]=df_test[col].fillna(method='bfill')
    #df[col]=df[col].fillna(df[col].value_counts().index[0])
    #df[col]=df[col].fillna(dtrain[col])
    #df_test[col]=df_test[col].fillna(dtest[col])

    """df['mail_type']=df['mail_type'].fillna(dtrain['mail_type'])
df['mail_category']=df['mail_category'].fillna(dtrain['mail_category'])
df_test['mail_type']=df_test['mail_type'].fillna(dtrain['mail_type'])
df_test['mail_category']=df_test['mail_category'].fillna(dtest['mail_category'])"""


# In[7]:

#features=['sent_time','last_online','hacker_created_at','hacker_timezone','contest_login_count','contest_login_count_1_days','contest_login_count_30_days','contest_login_count_365_days','contest_login_count_7_days','contest_participation_count','contest_participation_count_1_days','contest_participation_count_30_days','contest_participation_count_365_days','contest_participation_count_7_days','forum_comments_count','forum_count','forum_expert_count','forum_questions_count','hacker_confirmation','ipn_count','ipn_count_1_days','ipn_count_30_days','ipn_count_365_days','ipn_count_7_days','ipn_read','ipn_read_1_days','ipn_read_30_days','ipn_read_365_days','ipn_read_7_days','submissions_count','submissions_count_1_days','submissions_count_30_days','submissions_count_365_days','submissions_count_7_days','submissions_count_contest','submissions_count_contest_1_days','submissions_count_contest_30_days','submissions_count_contest_365_days','submissions_count_contest_7_days','submissions_count_master','submissions_count_master_1_days','submissions_count_master_30_days','submissions_count_master_365_days','submissions_count_master_7_days']
#for col in features:
#    df[col]=df[col].fillna(method='ffill')
#    df_test[col]=df_test[col].fillna(method='ffill')
    #df[col]=df[col].fillna(df[col].median())
    #df_test[col]=df_test[col].fillna(df_test[col].median())


# In[8]:

enc1 = LabelEncoder()
t_mail_train=enc1.fit_transform(df['mail_type'])
t_mail_test=enc1.transform(df_test['mail_type'])
df['mail_type_label']=t_mail_train
df_test['mail_type_label']=t_mail_test
enc2 = LabelEncoder()
t_cat_train=enc2.fit_transform(df['mail_category'])
t_cat_test=enc2.transform(df_test['mail_category'])
df['mail_category_label']=t_cat_train
df_test['mail_category_label']=t_cat_test

"""print len(df[df['mail_type'].isnull()])
print len(df[df['mail_category'].isnull()])
print len(df_test[df_test['mail_type'].isnull()])
print len(df_test[df_test['mail_category'].isnull()])"""


# In[9]:

features=['mail_category_label','mail_type_label','sent_time','last_online','hacker_created_at','hacker_timezone','contest_login_count','contest_login_count_1_days','contest_login_count_30_days','contest_login_count_365_days','contest_login_count_7_days','contest_participation_count','contest_participation_count_1_days','contest_participation_count_30_days','contest_participation_count_365_days','contest_participation_count_7_days','forum_comments_count','forum_count','forum_expert_count','forum_questions_count','hacker_confirmation','ipn_count','ipn_count_1_days','ipn_count_30_days','ipn_count_365_days','ipn_count_7_days','ipn_read','ipn_read_1_days','ipn_read_30_days','ipn_read_365_days','ipn_read_7_days','submissions_count','submissions_count_1_days','submissions_count_30_days','submissions_count_365_days','submissions_count_7_days','submissions_count_contest','submissions_count_contest_1_days','submissions_count_contest_30_days','submissions_count_contest_365_days','submissions_count_contest_7_days','submissions_count_master','submissions_count_master_1_days','submissions_count_master_30_days','submissions_count_master_365_days','submissions_count_master_7_days']
#features=['mail_category_label','sent_time','last_online','hacker_created_at','hacker_timezone','contest_login_count','contest_login_count_1_days','contest_login_count_30_days','contest_login_count_365_days','contest_login_count_7_days','contest_participation_count','contest_participation_count_1_days','contest_participation_count_30_days','contest_participation_count_365_days','contest_participation_count_7_days','forum_comments_count','forum_count','forum_expert_count','forum_questions_count','hacker_confirmation','ipn_count','ipn_count_1_days','ipn_count_30_days','ipn_count_365_days','ipn_count_7_days','ipn_read','ipn_read_1_days','ipn_read_30_days','ipn_read_365_days','ipn_read_7_days','submissions_count','submissions_count_1_days','submissions_count_30_days','submissions_count_365_days','submissions_count_7_days','submissions_count_contest','submissions_count_contest_1_days','submissions_count_contest_30_days','submissions_count_contest_365_days','submissions_count_contest_7_days','submissions_count_master','submissions_count_master_1_days','submissions_count_master_30_days','submissions_count_master_365_days','submissions_count_master_7_days']


# In[10]:

"""
df = df.fillna(df['mail_type'].value_counts().index[0])
df = df.fillna(df['mail_category'].value_counts().index[0])
df_test = df_test.fillna(df_test['mail_type'].value_counts().index[0])
df_test = df_test.fillna(df_test['mail_category'].value_counts().index[0])

df['mail_type']=df['mail_type'].fillna(method='ffill')
df['mail_category']=df['mail_category'].fillna(method='ffill')
df_test['mail_type']=df_test['mail_type'].fillna(method='ffill')
df_test['mail_category']=df_test['mail_category'].fillna(method='ffill')

print len(df[df['mail_type'].isnull()])
print len(df[df['mail_category'].isnull()])
print len(df_test[df_test['mail_type'].isnull()])
print len(df_test[df_test['mail_category'].isnull()])

"""


# In[11]:

#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import VotingClassifier

#from sklearn.tree import DecisionTreeClassifier


# In[ ]:

#model=ExtraTreesClassifier(n_estimators=100)
model=RandomForestClassifier(n_estimators=150)
#model=BaggingClassifier(base_estimator=RandomForestClassifier(),n_estimators=50)

#model=VotingClassifier(estimators=[('ad',AdaBoostClassifier(base_estimator=RandomForestClassifier(),n_estimators=5)),('rf', RandomForestClassifier(n_estimators=50)), ('et', ExtraTreesClassifier(n_estimators=50)), ('bg', BaggingClassifier(n_estimators=50))], voting='hard')
#model=GradientBoostingClassifier(n_estimators=150)
#model=AdaBoostClassifier(base_estimator=RandomForestClassifier(),n_estimators=15)

#model=DecisionTreeClassifier()

print(":::learning the model::: ")


# In[ ]:

model.fit(df[features],df['opened'])
print(":::predicting::: ")


# In[ ]:

predictions=model.predict(df_test[features])
"""
df_test['predictions']=predictions
for i,row in df_test.iterrows():
	if row['user_id'] in user_unsub:
		if row['mail_category'] in user_unsub[row['user_id']]:
			df_test.set_value(i,'predictions',False)
predictions=df_test['predictions']
"""

# In[ ]:

os.remove('predictions.csv')
f = open('predictions.csv','w')
for val in predictions:
    f.write(str(val*1)+'\n')    
f.close()


# In[ ]:

#save_object(df,"df.pkl")
#save_object(model,"model.pkl")


# In[3]:

#get_ipython().system(u'jupyter nbconvert --to script email_classifier.ipynb')


# In[ ]:



