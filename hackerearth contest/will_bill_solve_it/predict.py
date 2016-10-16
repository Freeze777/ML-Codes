
# coding: utf-8

# In[1]:

import pandas as pd
import seaborn as sb
from random import shuffle
import numpy as np
import pickle
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import os


# In[2]:

# For writing objects to files
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    return

def load_object(filename):
    with open(filename, 'rb') as input:
        obj=pickle.load(input)
    return obj


# In[3]:

#reading the csv files
df_prb=pd.read_csv("train/problems.csv")
df_sub=pd.read_csv("train/submissions.csv")
df_usr=pd.read_csv("train/users.csv")


# In[4]:

#group by user_id and problem_id to see submission status
cols=['user_id','problem_id']
df_grp=df_sub.groupby(cols)


# In[5]:

grp_ids=df_grp.groups


# In[6]:

user_problem_solved={}


# In[7]:

#target labels PAC=-1 and AC=+1
df_sub['result'].unique()


# In[8]:

#sample user_id-problem_id pair
df_grp.get_group( (1178898, 926073))


# In[9]:

#Computing the (user_id,problem_id,solved) dictionary where (user_id,problem_id) is the key solved(+1) or not solved(-1) is the value
#Uncomment to recompute user_problem_label object
"""user_problem_label={}
#test_grp=[(967552,909306),(1178898,926073),(1178898,926073),(1037442,916711),(1130935,913129)]
itr=0
for key in grp_ids:
#for key in test_grp:
    print "\ruser "+str(itr)+":"+str(key[0]),
    itr+=1
    tmp=df_grp.get_group(key)
    #check whether user has got the problem accepted
    if len(tmp[tmp['result']=='AC'])==0: # partially accepted
        user_problem_label[key]= -1
    else: 
         user_problem_label[key]= +1 #accepted"""



#creating a new csv file storing (user_id,problem_id,label) where label says solved or not solved
#uncomment to rewrite the csv file
"""f = open('train/user_problem_labels.csv','w')
f.write("user_id,problem_id,label\n")
for key in user_problem_label:
    to_write=str(key[0])+','+str(key[1])+','+str(user_problem_label[key])  
    f.write(to_write+'\n')    
f.close()"""


# In[12]:

#reading the custom made csv file
df_usr_prb=pd.read_csv("train/user_problem_labels.csv")


# In[13]:

df_usr_prb.head()


# In[14]:

#joining the train dataframes
df = pd.merge(df_prb, df_usr_prb, how='inner', on=['problem_id'])
df_train = pd.merge(df_usr, df, how='inner', on=['user_id'])


# In[15]:

#reading the test data
df_tst_prb=pd.read_csv("test/problems.csv")
df_tst_usr=pd.read_csv("test/users.csv")
df_pred=pd.read_csv("test/test.csv")


# In[16]:

#joining the test dataframes
df_t = pd.merge(df_tst_prb, df_pred, how='inner', on=['problem_id'])
df_tst = pd.merge(df_tst_usr, df_t, how='inner', on=['user_id'])


# In[17]:

df_tst.head(2)


# In[18]:

df_train.head(2)


# In[19]:

df_train.columns


# In[20]:

df_tst.columns


# In[21]:

#features I have decided to consider for now
features=['solved_count_x','attempts','user_type','level','accuracy','solved_count_y','error_count','rating','tag1','tag2']


# In[22]:

#filling empty cell in both test and train using forward fill approach.
# could have tried mode too..!!
for col in features:
    df_train[col]=df_train[col].fillna(method='ffill')
    df_tst[col]=df_tst[col].fillna(method='ffill')


# In[23]:

#df_train.head(10)
#Identiying categorical inputs
categorical=['user_type','level','tag1','tag2']


# In[24]:

df_train.describe()


# In[25]:

#Label encoding categorical inputs so that classifier can use them
from sklearn.preprocessing import LabelEncoder


# In[26]:

for col in categorical:
    enc = LabelEncoder()
    enc.fit(df_train[col].append(df_tst[col]))
    modified_train_col=enc.transform(df_train[col])
    modified_tst_col=enc.transform(df_tst[col])
    df_train[col+' enc']=modified_train_col
    df_tst[col+' enc']=modified_tst_col


# In[27]:

#processed feature columns
features=['solved_count_x','attempts','user_type enc','level enc','accuracy','solved_count_y','error_count','rating','tag1 enc','tag2 enc']


# In[34]:

#df_train[features].head()


# In[29]:

######## CLASSIFICATION #### 


#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier


# In[30]:

model=RandomForestClassifier(n_estimators=150)
#model=BaggingClassifier(n_estimators=150)
#model=DecisionTreeClassifier()
#model=ExtraTreesClassifier(n_estimators=150)


# In[31]:

print(":::learning the model::: ")

model.fit(df_train[features],df_train['label'])


# In[48]:

print(":::predicting for test data::: ")
predictions=model.predict(df_tst[features])
df_tst['prediction']=predictions
result = pd.DataFrame({        
        "prediction": df_tst['prediction'],        
        "problem_id": df_tst["problem_id"],
        "user_id": df_tst["user_id"],
    })
result.to_csv("test/predictions.csv", index=False,columns=['user_id','problem_id','prediction'])


# In[35]:

#Computing training error
train_predictions=model.predict(df_train[features])
df_train['prediction']=train_predictions


# In[37]:

df_train[['label','prediction']].head(10)


# In[44]:

#Finding accuracy
df_train['correct_prediction']=(df_train['label']==df_train['prediction'])


# In[45]:

df_train[['label','prediction','correct_prediction']].head(10)


# In[46]:

accuracy=sum(df_train['correct_prediction'])*100/float(len(df_train))


# In[50]:

print("Training accuracy (%)",accuracy)


# In[ ]:



