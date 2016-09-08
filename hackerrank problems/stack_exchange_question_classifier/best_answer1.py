# Enter your code here. Read input from STDIN. Print output to STDOUT
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

f = open('training.json','r')
N = int(f.readline())
i=0
train_ans=[]
train=[]
topic_list=[]
for line in f:
    j = json.loads(line)
    topic = (j['topic'])
    if topic in topic_list:
        topicN = topic_list.index(topic)
    else:
        topic_list.append(topic)
        topicN = topic_list.index(topic)
    train_ans.append(topicN)
    train_ans.append(topicN)
    train.append(j['question'])
    train.append(j['excerpt'])

test=[]
N = int(raw_input())
for i in range(N):
    j = json.loads(raw_input())
    test.append(j['question']+ " " + j['excerpt'])

    
v = TfidfVectorizer(sublinear_tf=True,analyzer = 'word', max_df = 0.1, ngram_range=(1,2),stop_words="english")
X = v.fit_transform(train)
Y = train_ans
X_test = v.transform(test)

clf = SGDClassifier(alpha=.00002, n_iter=50, penalty="l2")
clf.fit(X, Y)

predictions = clf.predict(X_test)
for i in predictions:
    print topic_list[i]