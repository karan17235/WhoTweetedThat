import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
nb = naive_bayes.MultinomialNB()

stop_words = set(stopwords.words('english'))

def process_tweet(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"handle", "", text)
    clean_text = []
    clean_text.append(text)
    return clean_text

def main():
        
    X = []
    # Y = []

    with open('DataSet/RawData/train_tweets.txt', 'r', encoding='utf-8') as tweet_file:
        # i = 0
        for row in tweet_file:
    #         row_values = ("".join(row)).split('\t')
            row_split = row.split('\t')
            X.append([row_split[0].strip(), "".join(process_tweet(row_split[1].strip()))])
            # Y.append("".join(process_tweet(row_split[1].strip())))
            # i = i + 1
            # if (i > 5):
            #     break

    # print("Value of X: ", X, "\n\n")

    # to_predict = []
    # with open('DataSet/TestData/test_tweets_unlabeled.txt', 'r', encoding='utf-8') as test_file:
    #     i = 0
    #     for row in test_file:
    #         to_predict.append("".join(process_tweet(row_split[1].strip())))
    #         i += 1
    #         if (i > 5):
    #             break

    # print("Prediction List: ", to_predict, "\n\n")

    data_frame = pd.DataFrame(X, columns = ['UserID', 'Tweet'], index=None)
    # print("Data Frame: ", data_frame, "\n\n")

    print("Splitting the Data Set: \n")
    # from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data_frame.Tweet, data_frame.UserID, random_state = 0, train_size = 0.3)

    encoder = LabelEncoder()

    print("Y Train Encoder: \n")
    y_train = encoder.fit_transform(y_train)
    print("Y Test Encoder: \n")
    y_test = encoder.fit_transform(y_test)

    print("Word Vectorizer: \n")
    # vectorizer = TfidfVectorizer(max_features = 2000, min_df = 5, max_df = 0.7)
    # train_tfidf = vectorizer.fit_transform(data_frame['Tweet'])
    # print("Vectorizer to Array: ", train_tfidf) #.toarray())

    wordVectorizer = CountVectorizer(stop_words = 'english', ngram_range = (1,2), min_df = 1, max_features = 15000)

    print("Character Vectorizer: \n")
    charVectorizer = CountVectorizer(analyzer='char', stop_words='english', ngram_range=(2,4), max_features=50000)

    print("Word Vectorizer Fit: \n")
    wordVectorizer.fit(pd.Series(data_frame['Tweet']))
    print("Character Vectorizer Fit: \n")
    charVectorizer.fit(pd.Series(data_frame['Tweet']))

    print("Word Vectorizer Transform Train: \n")
    XTrainWord = wordVectorizer.transform(X_train)
    print("Character Vectorizer Transform Train: \n")
    XTrainChar = charVectorizer.transform(X_train)

    print("Hstack Train: \n")
    XTrainTran = hstack([XTrainChar, XTrainWord])

    print("Word Vectorizer Transform Test: \n")
    XTestWord = wordVectorizer.transform(X_test)
    print("Word Vectorizer Transform Test: \n")
    XTestChar = charVectorizer.transform(X_test)

    print("Hstack Test: \n")
    XTestTran = hstack([XTestChar, XTestWord])
    
    print("Fit: \n")
    nb.fit(XTrainTran, y_train)

    print("Predict: \n")
    yPredClass = nb.predict(XTestTran)

    print('Accuracy: ', accuracy_score(y_test, yPredClass)*100)


    # train_tfidf = vectorizer.fit(data_frame['Tweet'])

    # XTrain_tfidf = vectorizer.transform(X_train)
    # XTest_tfidf = vectorizer.transform(X_test)

    # print("Vocabulary: ", vectorizer.vocabulary_, "\n")
    # print("X Train TFIDF: ", XTrain_tfidf, "\n\n")
    # print("X Test TFIDF: ", XTest_tfidf, "\n\n")

    # print("Fit the training dataset on the NB classifier:\n\n")
    # Naive = naive_bayes.MultinomialNB()
    # Naive.fit(XTrain_tfidf, y_train)

    # # Predict the labels on validation dataset
    # predictions_NB = Naive.predict(XTest_tfidf)

    # # Use accuracy_score function to get the accuracy
    # print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, y_test)*100)

main()
