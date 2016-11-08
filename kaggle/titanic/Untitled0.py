
import pandas as pd
import numpy as np

df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")


print "unsurvived ",(len(df_train)-len(df_train[df_train['Survived']==1]))/float(len(df_train))
#checking balance it is ok becoz 60 - 40 %
#if ratio is not good we make dem equal.. take smallest one..and sample other using that much size sample

df_train.columns

#checking for incomplete cells
for col in df_train.columns:
    print col,sum(df_train[col].isnull())


categorical_inc=['Embarked'] #have NA
#categorical_inc=['Embarked'] #have NA
for cat in categorical_inc:
    mode=df_train[cat].value_counts().idxmax()
    df_train[cat]=df_train[cat].fillna(mode)  
    df_test[cat]=df_test[cat].fillna(mode)  
    #print mode  
#cabin is imp we have to take 1 alphabet from cabin 

for col in df_train.columns:
    print col,sum(df_train[col].isnull())

#age is numerical : fill with meidan
#median_age=np.median(df_train['Age'])
df_train['Age']=df_train['Age'].fillna(df_train['Age'].median())   
df_test['Age']=df_test['Age'].fillna(df_train['Age'].median())   
df_train['Fare']=df_train['Fare'].fillna(df_train['Fare'].median())   
df_test['Fare']=df_test['Fare'].fillna(df_train['Fare'].median())   

df_train['Cabin'].unique()

df_train['Cabin']=df_train['Cabin'].fillna("#")
df_test['Cabin']=df_test['Cabin'].fillna("#")


#now data is clean
train_cabin=[]
test_cabin=[]
#print df_train.iterrows()
for i,row in df_train.iterrows():
    if(row['Cabin']!='#'):
        train_cabin.append(row['Cabin'][0])
    else:
        train_cabin.append('#')
for i,row in df_test.iterrows():
    if(row['Cabin']!='#'):
        test_cabin.append(row['Cabin'][0])
    else:
        test_cabin.append('#')
df_train['Cabin']=train_cabin
df_test['Cabin']=test_cabin

print sum([x=='#' for x in train_cabin ])
print len(train_cabin)
ss=set(train_cabin)

for val in ss:
    print val,sum([x==val for x in train_cabin ])

'''for row in train_cabin:
    if(row!='#'):
        df_train['Cabin']=row
    else:    
        df_train['Cabin']='C'
for row in test_cabin:
    if(row!='#'):
        df_test['Cabin']=row
    else:    
        df_test['Cabin']='C'
        '''





#since scikit learn classifier need numerical features ..we need label encoding for categorial
#categorical_feature=['Sex','Embraked']
#will do label encoding for that have to append test data also.
target=['Survived']
l=df_train['Cabin']
l.append(df_test['Cabin'])
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(l)
df_train['Cabin']=le.transform(df_train['Cabin'])
df_test['Cabin']=le.transform(df_test['Cabin'])
l=[]
embarked=[]
for i,row in df_train.iterrows():
    l.append(row['Sex'])
    embarked.append(row['Embarked'])
for i,row in df_test.iterrows():
    l.append(row['Sex'])    
    embarked.append(row['Embarked'])
le.fit(l)
df_train['Sex']=le.transform(df_train['Sex'])
df_test['Sex']=le.transform(df_test['Sex'])

le.fit(embarked)
df_train['Embarked']=le.transform(df_train['Embarked'])
df_test['Embarked']=le.transform(df_test['Embarked'])

#get label encoder from scikit learn
#use fit ... provide test and train
#use transform for both--->new columns in test and train
#print "trianing ",df_train[feature_need]
#print "Tetsing -->",df_test[feature_need]


import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv("train.csv")
fig = plt.figure(figsize=(18,4), dpi=1600)
alpha_level = 0.65
# building on the previous code, here we create an additional subset with in the gender subset 
# we created for the survived variable. I know, thats a lot of subsets. After we do that we call 
# value_counts() so it it can be easily plotted as a bar graph. this is repeated for each gender 
# class pair.
ax1=fig.add_subplot(141)
female_highclass = df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts()
female_highclass.plot(kind='bar', label='female, highclass', color='#FA2479', alpha=alpha_level)
ax1.set_xticklabels(["Survived", "Died"], rotation=0)
ax1.set_xlim(-1, len(female_highclass))
plt.title("Who Survived? with respect to Gender and Class"); plt.legend(loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
female_lowclass = df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts()
female_lowclass.plot(kind='bar', label='female, low class', color='pink', alpha=alpha_level)
ax2.set_xticklabels(["Died","Survived"], rotation=0)
ax2.set_xlim(-1, len(female_lowclass))
plt.legend(loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
male_lowclass = df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts()
male_lowclass.plot(kind='bar', label='male, low class',color='lightblue', alpha=alpha_level)
ax3.set_xticklabels(["Died","Survived"], rotation=0)
ax3.set_xlim(-1, len(male_lowclass))
plt.legend(loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
male_highclass = df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts()
male_highclass.plot(kind='bar', label='male, highclass', alpha=alpha_level, color='steelblue')
ax4.set_xticklabels(["Died","Survived"], rotation=0)
ax4.set_xlim(-1, len(male_highclass))
plt.legend(loc='best')




# we are for training model
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
feature_need=['Pclass','Sex','Age','Fare',#'Parch',
              'Cabin','Embarked']
#model= linear_model.Perceptron(fit_intercept=False, n_iter=100, shuffle=False)
model=linear_model.LogisticRegression(C=1e5)
#model=RandomForestClassifier(n_estimators=2)
model.fit(df_train[feature_need], df_train[target])
#print df_test[feature_need]

predictions= model.predict(df_test[feature_need])

#since we have lot of categorical inputs ---->use non-distance based classifier

#see training accuracy

#try xgboost 
#try seaborn for plotting

predictions_train= model.predict(df_train[feature_need])
#print target
#print sum(df_train[target].isnull())
print "error  ",sum(df_train['Survived']!=predictions_train)/float(len(df_train))

import pandas as pd
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("submission.csv", index=False)




