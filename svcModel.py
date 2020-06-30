# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

import string
from collections import Counter
import re
#import pattern3.en
from pattern3.en import spelling
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import unicodedata
#from spellchecker import SpellChecker
#spell = SpellChecker()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
stop_words = set(stopwords.words('english'))
lmt = WordNetLemmatizer()


#%%
np.random.seed(500)


#%%
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


#%%
X = []
y = []

with open('../data/train_tweets.txt') as tweet_f:
    
    for line in tweet_f:
        line_split = line.split('\t')
        
        X.append(process_tweet(line_split[1].strip()))
        y.append(line_split[0])


#%%
X_df= pd.DataFrame(X)
from sklearn.model_selection import train_test_split
train_size = 0.8

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=4)


#%%
Encoder = LabelEncoder()
Y_train = Encoder.fit_transform(Y_train)
Y_test = Encoder.fit_transform(Y_test)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(X)

print(Tfidf_vect.vocabulary_)


#%%
Train_X_Tfidf = Tfidf_vect.transform(X_train)
Test_X_Tfidf = Tfidf_vect.transform(X_test)


#%%
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Y_train)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Y_test)*100)


#%%
with open("svc_5000_model", 'wb') as file:
    pickle.dump(SVM, file)


