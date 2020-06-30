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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib

# svmSVC = svm.SVC()
lmt = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

all_words={}

def process_tweet(text):
    
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    normalised_tokens = []
    split_text = [re.sub(r"[^@a-zA-Z0-9']+", ' ', k).lower().strip() for k in text.split(" ")]
    for item in split_text:
        if item not in stop_words:
            normalised_item = lmt.lemmatize(item)
            if normalised_item in all_words:
                all_words[normalised_item]+=1
            else:
                all_words[normalised_item]=1
            normalised_tokens.append(item)
    normalised_tweet = " ".join([x for x in normalised_tokens])
    return normalised_tweet


def main():
    X = []
    y = []

    with open('DataSet/RawData/train_tweets.txt', 'r', encoding='utf-8') as tweet_f:
        
        for line in tweet_f:
            line_split = line.split('\t')
            
            X.append(process_tweet(line_split[1].strip()))
            y.append(line_split[0])

    X_unlable = []
    with open('DataSet/TestData/test_tweets_unlabeled.txt') as tweet_f:
        
        for line in tweet_f:
            X_unlable.append(process_tweet(line.strip()))



    # X_df= pd.DataFrame(X)
    # from sklearn.model_selection import train_test_split
    # train_size = 0.8

    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=4)

    Encoder = LabelEncoder()

    print("Encoder: \n")
    y = Encoder.fit_transform(y)
    # Y_test = Encoder.fit_transform(Y_test)

    print("Vectorizer: \n")
    count_vector = CountVectorizer(stop_words = 'english', ngram_range = (1,2), min_df = 1, max_features=5000)

    print("Fit: \n")
    count_vector.fit(X)

    # print(count_vector.vocabulary_)

    print("Transform: \n")
    Train_X_ctv = count_vector.transform(X)
    Test_X_ctv = count_vector.transform(X_unlable)

    print("Calling SVM SVC and Fitting: \n")
    SVM_model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM_model.fit(Train_X_ctv, y)
    # predict the labels on validation dataset

    print("Saving SVM Model: \n")
    joblib.dump(SVM_model, 'Models/aug_4_SVM_SVC.pkl')

    print("Predict: \n")
    predictions_SVM = SVM_model.predict(Test_X_ctv)
    # Use accuracy_score function to get the accuracy
    # print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Y_test)*100)

    df = pd.DataFrame({"Id":range(1, len(predictions_SVM) + 1 ,1), "Predicted":predictions_SVM})
    df.to_csv('Predictions/aug_4_SVM_SVC.csv', index=False)

main()