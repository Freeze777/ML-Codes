import sframe
import string
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import metrics
def remove_punctuation(text):
    return text.translate(None, string.punctuation)

products = sframe.SFrame('amazon_baby.gl/')
products = products.fillna('review',{'review':''})  # fill in N/A's in the review column
products = products[products['rating'] != 3]
products['review_clean'] = products['review'].apply(remove_punctuation) #creates a new column
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

#print products
#products.print_rows(num_rows=1)
#products.print_rows(num_rows=1,num_columns=4)
#print products[products['sentiment']==-1]


train_data, test_data = products.random_split(.8, seed=1)

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping

test_matrix = vectorizer.transform(test_data['review_clean'])
#print train_matrix.shape
log_reg=linear_model.LogisticRegression()

log_reg.fit(train_matrix,train_data['sentiment'])

test_result=predict(test_matrix)

accrcy=metrics.accuracy_score(test_data['sentiment'],test_result)