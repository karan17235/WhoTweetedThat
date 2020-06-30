import re
import pandas as pd
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
import joblib
from nltk.corpus import stopwords


df = pd.read_csv('DataSet/RawData/train_tweets.txt', sep="\t", header=None, names=["id", "tweet"])

# Loading the test dataset
df_test = pd.read_csv('DataSet/TestData/test_tweets_unlabeled.txt', sep="\t", header = None, names=["test_tweet"])
# print(type(df_test))

def word_count(sentence):
    return len(sentence.split())
    
df['word_count'] = df['tweet'].apply(word_count)

stop_words = set(stopwords.words('english'))
def processTweet(tweet):
    # print("Tweet: ", tweet)
    # To lowercase
    # tweet = ("".join(tweet[1:])).lower()
    # Removing http links from the tweet
    tweet = re.sub(r"http\S+", "", tweet)
    # Removing 'handle' keyword from the tweets
    tweet = re.sub(r"handle", "", tweet)
    # To lowercase
    tweet = tweet.lower()
    # tweet = (lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    tweet = ''.join(c for c in tweet if c <= '\uFFFF') 
    return tweet

# clean dataframe's text column
# df['tweet'] = df['tweet'].apply(processTweet)
# preview some cleaned tweets
# df['tweet'].head()

# CLeaning the test dataframe
df_test['test_tweet'] = df_test['test_tweet'].apply(processTweet)
# df['test_tweet'].head()

all_words = []
for line in list(df['tweet']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())
Counter(all_words).most_common(10)

# def remove_words(word_list):
#     remove = ['paul','ryan','...','“','”','’','…','ryan’']
#     return [w for w in word_list if w not in remove]
# -------------------------------------------
# tokenize message column and create a column for tokens

# vectorizer
print("Vectorizer Un-labled: \n")
count_vector = CountVectorizer(strip_accents='ascii', stop_words='english', lowercase=True, max_features=2000)  # Un labled
# transform the entire DataFrame of messages
count_vector.fit(df_test["test_tweet"])
test_unlable = count_vector.transform(df_test['test_tweet'])    # Un labled

print("Train Test split: \n")
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['id'], test_size=0.2)

# create pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(strip_accents='ascii', stop_words='english', lowercase=True, max_features=2000)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),
    ('classifier', svm.SVC(gamma='auto')),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

# this is where we define the values for GridSearchCV to iterate over
parameters = {'bow__ngram_range': [(1, 1), (1, 2)], 'classifier__kernel': ('linear', 'rbf'),}

# do 10-fold cross validation for each of the 6 possible combinations of the above params
print("Pipeline: \n")
grid = GridSearchCV(pipeline, cv=5, param_grid=parameters, verbose=1)

print("Fit: \n")
grid.fit(X_train, y_train)

# summarize results
print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
print('\n')
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))

# save best model to current working directory
print("Dump Model: \n")
joblib.dump(grid, "Models/aug_5_SVM_SVC_Grid.pkl")

# load from file and predict using the best configs found in the CV step
print("Load Model: \n")
model_SVM = joblib.load("Models/aug_5_SVM_SVC_Grid.pkl" )

# get predictions from best model above
print("Predict TestSplit: \n")
y_preds = model_SVM.predict(X_test)
print("Accuracy score on TestSplit: ", accuracy_score(y_test, y_preds))
print('\n')

print("Predict Un-labled: \n")
unlable_pred = model_SVM.predict(test_unlable)
df = pd.DataFrame({"Id":range(1, len(unlable_pred) + 1 ,1), "Predicted":unlable_pred})
df.to_csv("Predictions/aug_5_SVM_SVC_Grid.csv", index=False)