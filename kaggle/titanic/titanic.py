
# coding: utf-8

# In[1]:

import pandas as pd
import xgboost as xgb

# In[23]:

titanic_train=pd.read_csv('train.csv')


# In[24]:

desc=titanic_train.describe()#only shows numeric columns
print desc
#print desc.columns.values
#print titanic_train.columns.values
incomplete_columns=[col for col in desc.columns.values if desc[col]['count']!=len(titanic_train)]
print incomplete_columns


# In[25]:

for col in incomplete_columns:
    titanic_train[col]=titanic_train[col].fillna(titanic_train[col].median())
print(titanic_train.describe())


# In[37]:

cols_desc=set(desc.columns.values)
cols_train=set(titanic_train.columns.values)
cols_train.difference_update(cols_desc)
non_numeric_cols=list(cols_train)
print non_numeric_cols


# In[39]:

titanic_train['Sex_num']=titanic_train['Sex'].apply(lambda sex:0 if sex=='male' else 1)


# In[40]:

titanic_train.drop(['Sex'],inplace=True,axis=1)


# In[47]:

print(titanic_train['Embarked'].unique())
most_freq_embark=titanic_train['Embarked'].value_counts().idxmax()
titanic_train['Embarked']=titanic_train['Embarked'].fillna('S')


# In[49]:

titanic_train['Embarked_num']=titanic_train['Embarked'].apply(lambda sex:0 if sex=='S' else (1 if sex=='C' else 2))


# In[51]:

titanic_train.drop(['Embarked'],inplace=True,axis=1)


# In[58]:

print titanic_train.describe()


# In[59]:

features=['Pclass','Age','SibSp','Parch','Fare','Sex_num','Embarked_num']


# In[60]:

#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[61]:

#model=LogisticRegression()
#model=AdaBoostClassifier()
#model=RandomForestClassifier();
#model=DecisionTreeClassifier()
model = xgb.XGBClassifier(max_depth=5, n_estimators=320, learning_rate=0.05)
model.fit(titanic_train[features],titanic_train['Survived'])
#predictions = model.predict(test_X)



# In[77]:

titanic_test=pd.read_csv('test.csv')


# In[78]:

desc=titanic_test.describe()#only shows numeric columns
incomplete_columns=[col for col in desc.columns.values if desc[col]['count']!=len(titanic_test)]
print incomplete_columns


# In[79]:

for col in incomplete_columns:
    titanic_test[col]=titanic_test[col].fillna(titanic_test[col].median())
print(titanic_test.describe())


# In[80]:

titanic_test['Age']=titanic_test['Age'].fillna(titanic_test['Age'].median())
titanic_test['Sex_num']=titanic_test['Sex'].apply(lambda sex:0 if sex=='male' else 1)
titanic_test['Embarked']=titanic_test['Embarked'].fillna('S')
titanic_test['Embarked_num']=titanic_test['Embarked'].apply(lambda sex:0 if sex=='S' else (1 if sex=='C' else 2))
titanic_test.drop(['Sex','Embarked'],inplace=True,axis=1)


# In[81]:

predictions=model.predict(titanic_test[features])


# In[85]:

titanic_test['predictions']=predictions


# In[88]:

submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })


# In[89]:

submission.to_csv("submission.csv", index=False)


# In[ ]:



