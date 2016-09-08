
# coding: utf-8

# In[1]:

import pandas as pd
from random import shuffle
import numpy as np
import pickle


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

df_ch=pd.read_csv("challenges.csv")
df_sub=pd.read_csv("submissions.csv")


# In[3]:

print df_ch.columns
print df_sub.columns


# In[4]:

#(challenge_id,contest_id) is pk
df = pd.merge(df_ch, df_sub, how='inner', on=['challenge_id', 'contest_id'])


# In[ ]:
#####Buggy part in the code: idxmap{} was dependent on the original ordering of df######

#df=df.reindex(np.random.permutation(df.index))


# In[5]:

print df.columns


# In[6]:

cols=['challenge_id', 'contest_id', 'domain', 'subdomain', 'difficulty',
       'solved_submission_count', 'total_submissions_count', 'hacker_id',
       'language', 'solved', 'created_at']


# In[7]:

for col in cols:
    #df[col]=df[col].fillna(df[col].value_counts().index[0])
    df[col]=df[col].fillna(method='ffill')
    #print col,sum(df[col].isnull())


# In[8]:
#### Biggest bug: should have not used df['subdomain'].unique() instead should have used sorted(df['subdomain'].unique())#####
idxmap={}
idx=0
for col in df['subdomain'].unique():
    idxmap[idx]=col
    idx+=1


# In[13]:

hpref=load_object("hpref.pkl")
sdf_desc=load_object("sdf_desc.pkl")
sdf=load_object("sdf.pkl")


# In[14]:

mode_sub_dom=df["subdomain"].value_counts().index[0]
print mode_sub_dom


# In[16]:

df.sort(['solved_submission_count','total_submissions_count'], ascending=[False, False], inplace=True)


# In[17]:

subdom={}
for col in df['subdomain'].unique():
    subdom[col]=df.loc[df['subdomain']==col]
    


# In[19]:

hacker_solved=load_object("hacker_solved.pkl")
print len(hacker_solved)


# In[20]:

#print subdom['Dynamic Programming'].head()


# In[21]:

#l=hpref["00004cf8b853ad0d"]
#print idxmap[l.index(max(l))] #sorting


# In[22]:

#to do : before adding challenge to set check whether hacker has done it
print "computing recommendations"
rec={}
iter=0
hackers=df['hacker_id'].unique()
#hackers=["00004cf8b853ad0d","3827969344861ac8","b630307ea7151c3a"]
for hacker in hackers:
    print "\ruser "+str(iter)+":"+hacker,
    l=hpref[hacker]
    pref=[]
    for i in range(len(l)) :
        #if l[i]!=0:
        pref.append((l[i],i))
    pref.sort(reverse=True)
    hrec=set()
    size=4
    num=12
    for t in pref:
        tdf=subdom[idxmap[t[1]]]
        if len(hrec)>=num:
                break
        count=0
        for ii,row in tdf.iterrows():
            #if row['hacker_id']!=hacker and row['contest_id']=='c8ff662c97d345d2':
            if ((row['hacker_id']!=hacker) or (row['hacker_id']==hacker and row['solved']==0)) and row['contest_id']=='c8ff662c97d345d2':
                if row['challenge_id'] not in hrec:
                    if hacker not in hacker_solved:
                        hrec.add(row['challenge_id'])
                        ss=set()
                        ss.add(row['challenge_id'])
                        hacker_solved[hacker]=ss
                        count+=1
                    elif row['challenge_id'] not in hacker_solved[hacker]:
                        hrec.add(row['challenge_id'])
                        hacker_solved[hacker].add(row['challenge_id'])
                        count+=1
                    if count==size:
                        break
            if len(hrec)>=num:
                break
        #print idxmap[t[1]],len(hrec)
    l=list(hrec)
    shuffle(l)
    rec[hacker]=l
    iter+=1


# In[23]:

import os
os.remove("recommendation_.csv")
f = open('recommendation_.csv','w')
for key in rec:
    ans=''
    l=rec[key]
    l=l[0:10]
    for c in l:
        ans+=','+c    
    f.write(key+ans+'\n')    
f.close()


# In[1]:

#get_ipython().system(u'jupyter nbconvert --to script email_classifier.ipynb')


# In[ ]:



