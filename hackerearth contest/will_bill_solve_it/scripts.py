
# coding: utf-8

# In[1]:

import pandas as pd
import seaborn as sb
from random import shuffle
import numpy as np
import pickle
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


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

df_prb.columns


# In[5]:

df_usr.columns


# In[6]:

df_sub.columns


# In[7]:

df_usr.describe()


# In[8]:

df_sub.describe()


# In[9]:

df_prb.describe()


# In[10]:

users=df_usr['user_id'].unique()


# In[11]:

#average attempt for a user to solve a problem
tot_attempts=0
for ii,row in df_usr.iterrows():
    tot_attempts+=row['attempts']
print("Avg attempt for a user",tot_attempts/float(len(users)))


# In[12]:

df_prb.head(5)


# In[13]:

#checking unique values for level column
df_prb['level'].unique()


# In[14]:

#checking for incomplete cells in problem.csv
sum(df_prb['level'].isnull())


# In[15]:

#dropping null valued rows
df_prb_clean=df_prb.dropna(subset=['level'])


# In[16]:

len(df_prb_clean)


# In[17]:

#finding average accuracy for problems per level
level_avg_accuracy={}
level_count={}
for ii,row in df_prb_clean.iterrows():
    if row['level'] not in level_avg_accuracy:
        level_avg_accuracy[row['level']]=0
        level_count[row['level']]=0
    level_avg_accuracy[row['level']]+=row['accuracy']
    level_count[row['level']]+=1

for key in level_count:
    level_avg_accuracy[key]/=level_count[key]


# In[18]:

#Average accuracy per level
level_avg_accuracy


# **Plotting average accuracy for each levels of questions**

# In[19]:

sb.barplot(level_avg_accuracy.keys(),level_avg_accuracy.values())


# In[20]:

df_sub.head(5)


# In[21]:

#groupby user_id and problem_id to see submission streak
cols=['user_id','problem_id']
df_grp=df_sub.groupby(cols)


# In[22]:

#list of unique user_id-problem_id pair
grp_ids=df_grp.groups


# In[91]:

#printing 5 sample groups
i=0
for key in grp_ids:
    print(df_grp.get_group(key))
    if i>5:
        break
    i+=1


# In[24]:

df_tmp=df_grp.get_group((967552,909306 ))
#df_tmp=df_grp.get_group((1178898,926073))
#printlen(df_tmp[df_tmp['result']=='AC'])
#print (((df_tmp[df_tmp['result']=='AC']).head(1))['execution_time']).values[0]


# In[27]:

#code for computing users average percentage time improvements
#uncomment for recomputing user_time_imprv object
"""user_time_imprv={}
user_time_count={}
test_grp=[(967552,909306),(1178898,926073),(1178898,926073),(1037442,916711),(1130935,913129)]
itr=0
for key in grp_ids:
#for key in test_grp:
    print "\ruser "+str(itr)+":"+str(key[0]),
    itr+=1
    tmp=df_grp.get_group(key)
    if len(tmp[tmp['result']=='AC'])==0:
        continue
    else:
        best_exec_time=((tmp[tmp['result']=='AC']).head(1)['execution_time']).values[0]
    
    partial=tmp[tmp['result']=='PAC']
    if len(partial)==0:
        continue
    res=0
    count=0
    for ii,row in partial.iterrows():
        run_time=row['execution_time']
        if best_exec_time<run_time:
            res+=((run_time-best_exec_time)*100.0/best_exec_time)
            count+=1
    if count==0:
        continue
    #print   
    avg_res=res/count
    if key[0] not in user_time_imprv:
        user_time_imprv[key[0]]=0
    if key[0] not in user_time_count:
        user_time_count[key[0]]=0
    user_time_imprv[key[0]]+=avg_res
    user_time_count[key[0]]+=1
    
    
for user in user_time_imprv:
    user_time_imprv[user]/=user_time_count[user]"""


# In[28]:

#writing computational intensive code snippets to file
#save_object(user_time_imprv, "user_time_imprv.pkl")


# In[ ]:

#loading pickled file into python object
user_time_imprv=load_object("user_time_imprv.pkl")


# In[29]:

#user_time_imprv[967552]
#user_time_imprv[1178898]
#user_time_imprv[1130935]


# In[84]:

usr=[967552,1178898,1130935,1034752]
print("user_id","% improvement in execution time")
for key in usr:
    print(key,user_time_imprv[key])


# In[31]:

#why am in seeing huge improvement????
## there are cases where the runtime is 105 for PAC and 5.18 for AC
#df_sub[df_sub['user_id']==1130935]


# In[87]:

global_avg_time_imprv=0
for key in user_time_imprv:
    global_avg_time_imprv+=(user_time_imprv[key])/len(user_time_imprv)
print global_avg_time_imprv


# In[86]:

print len(user_time_imprv)


# In[88]:

#df_prb['tag1'].unique()


# In[39]:

#removing rows with empty tag1 values
df_prb_tag1_clean=df_prb.dropna(subset=['tag1'])


# In[45]:

len(df_prb_tag1_clean)


# In[44]:

#Computing the solved count for each tag1 category of questions
tag1_solved={}

for ii,row in df_prb_tag1_clean.iterrows():
    if row['tag1'] not in tag1_solved:
        tag1_solved[row['tag1']]=0
    tag1_solved[row['tag1']]+=row['solved_count']


# In[60]:

#which category of question are more solved
#sb.boxplot(tag1_solved.keys(),tag1_solved.values())
sb.barplot(tag1_solved.values()[0:28],tag1_solved.keys()[0:28])


# In[89]:

fig=sb.barplot(tag1_solved.values()[28:],tag1_solved.keys()[28:])


# In[66]:

#removing rows with empty tag2 column values
df_prb_tag2_clean=df_prb.dropna(subset=['tag2'])


# In[90]:

len(df_prb_tag2_clean)
#df_prb_tag2_clean.head()


# In[68]:

#computing the error count for each tag2 category of questions
tag2_error={}
for ii,row in df_prb_tag2_clean.iterrows():
    if row['tag2'] not in tag2_error:
        tag2_error[row['tag2']]=0
    tag2_error[row['tag2']]+=row['error_count']


# In[72]:

sb.barplot(tag2_error.values()[0:28],tag2_error.keys()[0:28])


# In[73]:

sb.barplot(tag2_error.values()[28:],tag2_error.keys()[28:])


# In[79]:

#Computing the total submissions count for each language

popular_language={}
for ii,row in df_sub.iterrows():
    if row['language_used'] not in popular_language:
        popular_language[row['language_used']]=0
    popular_language[row['language_used']]+=1
    


# In[81]:

sb.barplot(popular_language.values(),popular_language.keys())

