import pandas as pd
import numpy as np
import string
from collections import Counter
import re

from pattern3.en import spelling
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import unicodedata
# from spellchecker import SpellChecker
# spell = SpellChecker()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
stop_words = set(stopwords.words('english'))
lmt = WordNetLemmatizer()

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from collections import Counter

def get_handle_count(tweet):
    count=0
    for item in tweet.split(" "):
        if item=="@handle":
            count+=1
    return count

def get_hashtag_count(tweet):
    count=0
    for item in tweet.split(" "):
        if item[0]=="#":
            count+=1
    return count


def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

# def get_error_count(tweet):
#     count=0
#     for item in re.sub(r"[^a-zA-Z0-9']+", '', tweet).lower().split(" "):
#         if spell.correction(item)!=item:
#             count +=1
#     return count

def get_elongated_count(tweet):
    red = re.findall(r"(.)\1{2,}",tweet)
    return len(red)

def get_url_count(tweet):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    return len(urls)

def get_quotation_count(tweet):
    count=0
    for c in tweet:   
        if c == '''"''':
            count+=1
    return count

def get_word_count(tweet):
    return len(tweet.split(" "))

def get_sentence_count(tweet):
    return len(tweet.split("."))
    

def get_capitalised_word_count(tweet):
    count=0
    for item in tweet.split(" "):
        if item.isupper() and len(item)>1:
            count+=1
    return count

def get_punctuation_count(tweet):
    count=0
    for c in tweet:   
        if c in string.punctuation:
            count+=1
    return count


#def get_special_char_count(tweet):

def get_tweet_length(tweet):
    return len(tweet)

def process_tweet(text):
    '''
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    normalised_tokens = []
    split_text = [re.sub(r"[^a-zA-Z0-9']+", ' ', k).lower().strip() for k in text.split(" ")]
    for item in split_text:
        if item not in stop_words:
            normalised_item = lmt.lemmatize(item)
            if normalised_item in all_words:
                all_words[normalised_item]+=1
            else:
                all_words[normalised_item]=1
            normalised_tokens.append(item)
    '''

    output = []
    output.append(get_handle_count(text))
    output.append(get_hashtag_count(text))
    #output.append(get_error_count(text))
    output.append(get_elongated_count(text))
    output.append(get_url_count(text))
    output.append(get_quotation_count(text))
    output.append(get_word_count(text))
    output.append(get_sentence_count(text))
    output.append(get_capitalised_word_count(text))
    output.append(get_punctuation_count(text))
    
    
    return output

X = []
y = []

with open('DataSet/RawData/train_tweets.txt', 'r', encoding='utf-8') as tweet_f:

    for line in tweet_f:
        line_split = line.split('\t')
        
        X.append(process_tweet(line_split[1].strip()))
        y.append(line_split[0])
        
X_pred = []
with open('DataSet/TestData/test_tweets_unlabeled.txt', 'r', encoding='utf-8') as tweet_f:

    for line in tweet_f:
        X_pred.append(process_tweet(line.strip()))
#X= pd.DataFrame(X)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4000, input_dim=9, activation='relu'))
	model.add(Dense(9268, activation='sigmoid')) #The number of all possible labels
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#removing ys with less than 2

coll = Counter(y)

to_remove =[]
for item in coll:
    if coll[item]<2:
        to_remove.append(item)
#sorted(coll.items(), key=lambda pair: pair[1])
X_after = []
y_after = []
for i in range(len(y)):
    if y[i] not in to_remove:
        X_after.append(X[i])
        y_after.append(y[i])
        
print(len(y_after))
print(len(y))
        
X=X_after
y=y_after

coll = Counter(y)
len(coll)

X_norm = preprocessing.normalize(X)
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
#print(encoded_Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)

#Neural net start
seed = 7
numpy.random.seed(seed)
estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=100, verbose=1)

#kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
kfold = StratifiedShuffleSplit(n_splits=10, test_size = 0.1, random_state=seed)

results = cross_val_score(estimator, X_norm, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

estimator.fit(X,dummy_y)

model_json = estimator.model.to_json()
with open("Models/KerasWithFeatures.json", "w", encoding='utf-8') as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
estimator.model.save("Models/KerasWithFeatures.h5")

print("Model saved to disc")
