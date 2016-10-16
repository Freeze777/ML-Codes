import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation

# Load the data 
# dataset available at kaggle https://www.kaggle.com/c/titanic
train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)

# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

# feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']
feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch','SibSp','Ticket','Cabin','Embarked']
nonnumeric_columns = ['Sex','Ticket','Cabin','Embarked']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

# XGBoost doesn't (yet) handle categorical features automatically, so we need to change
# them to columns of integer values.
# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
# details and options
le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# Prepare the inputs for the model
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Survived']


### XGboost Baseline
gbm_baseline_clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

### XGboost
gbm_clf = xgb.XGBClassifier(max_depth=5, n_estimators=320, learning_rate=0.05)

### SVM
from sklearn import svm
svm_clf = svm.SVC(gamma=0.005, C=200)

### Random forest
from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier() #default 
rf = RandomForestClassifier(n_estimators=500)

# This function train classifiers N times. Each time it split training set for train/learn, 
# then fit classifier on training set and calculate MSE on test set
mse = [[],[],[],[]] # we will create 4 predictions
accuracy = [[],[],[],[]]
def do_folds(folds = 2):
    for f in range(0, folds):
        ### Split
        
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_X,
                                                            train_y, test_size=0.10)
        gbm_baseline = gbm_baseline_clf.fit(X_train, y_train)
        gbm_baseline_predictions = gbm_baseline.predict(X_test)

        gbm = gbm_clf.fit(X_train, y_train)
        gbm_predictions = gbm.predict(X_test)

        svm_clf.fit(X_train, y_train)
        svm_clf_predictions = svm_clf.predict(X_test)

        rf.fit(X_train, y_train)
        rf_clf_predict = rf.predict(X_test)
        
        mse[0].append(mean_squared_error(gbm_baseline_predictions, y_test))
        mse[1].append(mean_squared_error(gbm_predictions, y_test))
        mse[2].append(mean_squared_error(svm_clf_predictions, y_test))
        mse[3].append(mean_squared_error(rf_clf_predict, y_test))
        
        accuracy[0].append(accuracy_score(y_test, gbm_baseline_predictions))
        accuracy[1].append(accuracy_score(y_test, gbm_predictions))        
        accuracy[2].append(accuracy_score(y_test, svm_clf_predictions))        
        accuracy[3].append(accuracy_score(y_test, rf_clf_predict))        


#we learn wour models 50 time and cal MSE each time
do_folds(50) 

# Show accuracy for each prediction AND plot data
print np.mean(accuracy, axis=1)
print np.std(accuracy, axis=1)
accuracy0 = sns.distplot(accuracy[0],kde=False, rug=True, color='r', bins=20)
plt.show(accuracy0)
accuracy1 = sns.distplot(accuracy[1],kde=False, rug=True, color='r', bins=20)
plt.show(accuracy1)
accuracy2 = sns.distplot(accuracy[2],kde=False, rug=True, color='b', bins=20)
plt.show(accuracy2)
accuracy3 = sns.distplot(accuracy[3],kde=False, rug=True, color='g', bins=20)
plt.show(accuracy3)

#Show means for each prediction AND plot data
print np.mean(mse, axis=1)
mse0 = sns.distplot(mse[0],kde=False, rug=True, color='r', bins=20)
plt.show(mse0)
mse1 = sns.distplot(mse[1],kde=False, rug=True, color='r', bins=20)
plt.show(mse1)
mse2 = sns.distplot(mse[2],kde=False, rug=True, color='b', bins=20)
plt.show(mse2)
mse3 = sns.distplot(mse[3],kde=False, rug=True, color='g', bins=20)
plt.show(mse3)


# Store XGboost prediction resul
final_clf = gbm_baseline_clf.fit(train_X, train_y)
predictions = final_clf.predict(test_X)
# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)