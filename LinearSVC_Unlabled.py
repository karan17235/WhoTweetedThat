import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
import joblib

def main():
    X = []
    y = []
    
    with open('DataSet/RawData/train_tweets.txt', 'r', encoding='utf-8') as tweet_f:
        
        for line in tweet_f:
            line_split = line.split('\t')
            
            X.append(line_split[1].strip())
            y.append(line_split[0].strip())

    X_unlable = []
    with open('DataSet/TestData/test_tweets_unlabeled.txt', 'r', encoding='utf-8') as tweet_f:
        
        for line in tweet_f:
            X_unlable.append(line.strip())

    Encoder = LabelEncoder()

    print("Encoder: \n")
    y = Encoder.fit_transform(y)
    # Y_test = Encoder.fit_transform(Y_test)

    print("Vectorizer: \n")
    count_vector = CountVectorizer()

    print("Fit: \n")
    count_vector.fit(X)

    # print(count_vector.vocabulary_)

    print("Transform: \n")
    Train_X_ctv = count_vector.transform(X)
    Test_X_ctv = count_vector.transform(X_unlable)

    print("Calling SVM SVC and Fitting: \n")
    SVM_model = LinearSVC()
    SVM_model.fit(Train_X_ctv, y)
    # predict the labels on validation dataset

    print("Saving SVM Model: \n")
    joblib.dump(SVM_model, 'Models/Aug6_LinearSVC.pkl')

    print("Predict: \n")
    predictions_SVM = SVM_model.predict(Test_X_ctv)
    # Use accuracy_score function to get the accuracy
    # print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Y_test)*100)

    df = pd.DataFrame({"Id":range(1, len(predictions_SVM) + 1 ,1), "Predicted":predictions_SVM})
    df.to_csv('Predictions/Aug6_LinearSVC.csv', index=False)

main()
