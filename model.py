import pandas
import numpy
import tensorflow as tf
import joblib

from keras.models import Sequential, save_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dropout
from keras.preprocessing import text, sequence

max_words = 2000
num_classes = 0

def main():
    loadDataSet()
    # baselineModel()

def loadDataSet():
    tempFrame = []
    with open("DataSet/ProcessedData/ProcessedTrainSet.txt", "r", encoding='utf-8') as input_data:
        for line in input_data:
            row = []
            line = line.replace("\t", " ")
            tokens = line.strip().split(" ")
            row.append(tokens[0])
            row.append(" ".join(tokens[1:]))
            tempFrame.append(row)

    user_tweet = pandas.DataFrame(tempFrame, columns = ["Users", "Tweets"])
    X_train, X_test, y_train, y_test = train_test_split(user_tweet.Tweets, user_tweet.Users, random_state = 0, test_size = 0.3)

    print(X_train.shape, y_train.shape)
    print (X_test.shape, y_test.shape)

#     print("Data Split")

#     tokenize = text.Tokenizer(num_words=max_words, char_level=False)

#     tokenize.fit_on_texts(pandas.Series(user_tweet['Tweets'])) # only fit on train
#     x_train = tokenize.texts_to_matrix(X_train)
#     x_test = tokenize.texts_to_matrix(X_test)

#     print("Data tokenized")

#     encoder = LabelEncoder()
#     encoder.fit(pandas.Series(user_tweet['Users']))
#     y_train = encoder.transform(y_train)
#     y_test = encoder.transform(y_test)

#     print("Data Encoded")

#     num_classes = numpy.max(y_train) + 1

#     print("Model being trained")

#     # estimator = KerasClassifier(build_fn=baselineModel, epochs=50, batch_size=30, verbose=0)
#     # save_model(estimator, 'Models/kerasModel.h5', overwrite=True)
#     # kfold = KFold(n_splits=10, shuffle=True)
#     # results = cross_val_score(estimator, x_train, y_train, cv=kfold)
#     # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# # def baselineModel():
#     model = Sequential()
#     model.add(Dense(512, input_shape=(max_words,), activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))
#     #Compile model
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(x_train, y_train)
#     save_model(model, 'Models/kerasModel.h5', overwrite=True)
#     # return model

main()
