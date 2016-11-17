import pickle
import re
import string
import numpy as np
from nltk.stem.porter import PorterStemmer
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

def encode_onehot(df, cols):
    vec = DictVectorizer()

    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    return

def load_object(filename):
    with open(filename, 'rb') as input:
        obj=pickle.load(input)
    return obj


def balance_data(df,col_name,label1,label2):
	df_label1=df.loc[df[col_name]==label1]
	df_label2=df.loc[df[col_name]==label2]
	percentage = len(df_label1)/float(len(df_label2))
	if(percentage<=1.0):
		df_label2=df_label2.sample(frac=percentage)
		df=df_label1.append(df_label2)
	else:
		df_label1=df_label1.sample(frac=(1.0/percentage))
		df=df_label2.append(df_label1)
	df=df.reindex(np.random.permutation(df.index))
	return df

def strip_punctuations(data):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = data.translate(replace_punctuation)
    text = re.sub(' +',' ',text)
    return text

def strip_html(data):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    p = re.compile(r'<.*?>')
    text = p.sub(' ', data).translate(replace_punctuation)
    return text

def stem_text(data):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(data)
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return " ".join(stemmed)
