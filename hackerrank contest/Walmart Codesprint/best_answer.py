import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
stemmer = PorterStemmer()
from bs4 import BeautifulSoup
import re
import random; random.seed(1)

def str_stem(s): 
    if isinstance(s, str):
        s = s.lower()
        s = s.replace("  "," ")
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s = s.lower()
        return s
    else:
        return "null"

def str_stem2(s): 
    if isinstance(s, str):
        s = s.lower()
        s = s.replace(">","> ")
        s = s.replace("<"," <")
        b = BeautifulSoup(s, "lxml")
        s = b.get_text(" ").strip()
        s = (" ").join([z for z in s.split(" ")])
        s = s.replace("  "," ")
        s = s.replace(","," ")
        s = s.replace("$"," ")
        s = s.replace("("," ")
        s = s.replace(")"," ")
        s = s.replace("?"," ")
        s = s.replace("-","") #no space
        s = s.replace(":","") #no space
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        s = s.replace("  "," ")
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s = s.replace("  "," ")
        s = s.lower()
        return s
    else:
        return "null"

def fBlender(list1, list2):
    list1 = str(list1).strip('[]').replace(' ','').split(',')
    list2 = str(list2).strip('[]').replace(' ','').split(',')
    list1 += list2
    list1 = list(set(list1))
    list1 = [x for x in list1 if len(x)>0 and x !=' ']
    return '[' + ','.join(list1) + ']'

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['xProduct Name','xProduct Long Description']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

c = ['Seller', 'Actors', 'Actual Color', 'Artist ID', 'Aspect Ratio', 'Color', 'Genre ID', 'ISBN', 'Item Class ID', 'Literary Genre', 'MPAA Rating', 'Product Long Description', 'Product Name', 'Product Short Description', 'Publisher', 'Recommended Location', 'Recommended Room', 'Recommended Use', 'Short Description', 'Synopsis', 'actual_color']
train = pd.read_table('train.tsv', usecols=c).fillna('')
y_train = pd.read_table('train.tsv', usecols=['tag']).fillna('')
n = len(train)
test = pd.read_table('test.tsv', usecols=c).fillna('')
id_test = pd.read_table('test.tsv', usecols=['item_id'])

y_train['tag'] = y_train['tag'].map(lambda x: str(x).strip('[]').replace(' ',''))
y_train = y_train['tag'].str.get_dummies(sep=',')

df_all = pd.concat((train, test), axis=0, ignore_index=True)

cat = ['Seller', 'Actors', 'Actual Color', 'Aspect Ratio', 'Color', 'Item Class ID', 'Literary Genre', 'MPAA Rating', 'Product Long Description', 'Product Name', 'Product Short Description', 'Publisher', 'Recommended Location', 'Recommended Room', 'Recommended Use', 'Short Description', 'Synopsis', 'actual_color']
for c in cat:
    if c in ['Product Name','Product Long Description']:
        df_all['x'+c] = df_all[c].map(lambda x:str_stem2(x))
    df_all[c] = df_all[c].map(lambda x:str_stem(x))
    df_all['words_of'+c] = df_all[c].map(lambda x:len(x.split()))
    df_all['len_of'+c] = df_all[c].map(lambda x:len(x))
    df_u = pd.unique(df_all[c].ravel())
    print(c, len(df_u))
    d={}; j = 0
    for s in df_u:
        d[str(s)]=j; j+=1
    df_all[c] = df_all[c].map(lambda x:d[str(x)])
df_all = df_all.replace({'': 0}, regex=True)
train = df_all.iloc[:n]
test = df_all.iloc[n:]

etr = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, verbose=0, random_state=1, warm_start=False)
tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_df=0.5, min_df=5)
tsvd = TruncatedSVD(n_components=100, random_state = 1)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='xProduct Name')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='xProduct Long Description')), ('tfidf2', tfidf), ('tsvd2', tsvd)]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 1.0,
                        'txt2': 1.0
                        },
                n_jobs = -1
            )),
        ('etr', etr)])

for fold_ in range(len(y_train.columns)):
    random.seed(fold_)
    model = clf.fit(train, y_train[y_train.columns[fold_]])
    y_pred = model.predict(test)
    
    y = pd.DataFrame(y_pred)
    y.columns = [y_train.columns[fold_]]
    id_test = pd.concat([id_test, y], axis=1)

df_pr = id_test[:]
train_preds = {}
for row in df_pr.values:
    row = list(row)
    id = row.pop(0)
    active = [j[1] for j in sorted([[row[i],y_train.columns[i]] for i in range(len(y_train.columns)) if row[i]>.5], reverse=True)][:5]
    if len(active)<1:
        active = [j[1] for j in sorted([[row[i],y_train.columns[i]] for i in range(len(y_train.columns)) if row[i]>.347], reverse=True)][:4]
        if len(active)<1:
            active = [j[1] for j in sorted([[row[i],y_train.columns[i]] for i in range(len(y_train.columns))  if row[i]>.258], reverse=True)][:2]
            if len(active)<1:
                active = [j[1] for j in sorted([[row[i],y_train.columns[i]] for i in range(len(y_train.columns))  if row[i]>.24], reverse=True)][:1]
    train_preds[id] = active

id_test['tag'] = id_test['item_id'].map(lambda x: '[' + ','.join(train_preds[x]) + ']')
sub1 = id_test[:]

id_test = pd.read_table('test.tsv', usecols=['item_id'])
etr = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, verbose=0, random_state=1, warm_start=False)
tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_df=0.7, min_df=5)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='xProduct Name')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='xProduct Long Description')), ('tfidf2', tfidf), ('tsvd2', tsvd)]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 1.0,
                        'txt2': 1.0
                        },
                n_jobs = -1
            )),
        ('etr', etr)])

model = clf.fit(train, y_train.values)
y_pred = model.predict(test)

y = pd.DataFrame(y_pred)
y.columns = y_train.columns

df_pr = pd.concat([id_test, y], axis=1)
train_preds = {}
for row in df_pr.values:
    row = list(row)
    id = row.pop(0)
    active = [j[1] for j in sorted([[row[i],y_train.columns[i]] for i in range(len(y_train.columns)) if row[i]>.343], reverse=True)][:5]
    if len(active)<1:
        active = [j[1] for j in sorted([[row[i],y_train.columns[i]] for i in range(len(y_train.columns)) if row[i]>.335], reverse=True)][:4]
        if len(active)<1:
            active = [j[1] for j in sorted([[row[i],y_train.columns[i]] for i in range(len(y_train.columns)) if row[i]>.26], reverse=True)][:3]
            if len(active)<1:
                active = [j[1] for j in sorted([[row[i],y_train.columns[i]] for i in range(len(y_train.columns))  if row[i]>.252], reverse=True)][:2]
    train_preds[id] = active

id_test['tag'] = id_test['item_id'].map(lambda x: '[' + ','.join(train_preds[x]) + ']')
sub2 = id_test[:]
sub2.columns = ['item_id','tag_']

df = pd.merge(sub1, sub2, how='left', on='item_id')
df['tag'] = df.apply(lambda r: fBlender(r['tag'],r['tag_']), axis=1)
df = df[sub1.columns]
df.to_csv('submission.tsv', sep='\t', index=False)
print("Done!")
