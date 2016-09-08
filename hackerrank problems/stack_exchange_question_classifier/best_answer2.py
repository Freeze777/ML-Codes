import json
import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

X_data = []
y_data = []

train = open('training.json', 'r')
N = int(train.readline())
for i in range(N):
    parsed = json.loads(train.readline().lower())
    X_data.append(parsed['question'].strip())
    X_data.append(parsed['excerpt'].strip())
    y_data.append(parsed['topic'].strip())
    y_data.append(parsed['topic'].strip())
train.close()

tfidf_ngrams = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", binary=False, min_df=2, max_df=0.2, stop_words='english', use_idf=True, sublinear_tf=True)
clf = RidgeClassifier(alpha=1.2, normalize=True)

pipeline = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])
pipeline.fit(X_data, y_data)

X_test = []

N = input()
for i in range(N):
    parsed = json.loads(raw_input().lower())
    X_test.append(parsed['question'].strip() + ' ' + parsed['excerpt'].strip())
res = pipeline.predict(X_test)
print '\n'.join(res)

    