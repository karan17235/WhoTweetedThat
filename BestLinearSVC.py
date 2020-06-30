# Working with 25% accuracy but overfitting

print("Started: \n")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
from sklearn.svm import SVC,LinearSVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

data = pd.read_csv('DataSet/DataCSV/all_data.csv')

y=data['label']
X= pd.DataFrame(data,columns=['sentence_count','word_count','hashtags','upper','mentions','elongated','char_count', 
'punctuation_count','commas', 'semicolons', 'exclamations', 'periods', 'questions', 'quotes', 'ellipses','title_word_count','random_cap','Links','contract_count', 
'emotion_count','number_count','retweet_count','alpha_num','sentiment','abbrev','swear'])
X=preprocessing.normalize(X)

dataTest = pd.read_csv('DataSet/DataCSV/all_test.csv')
print("Loaded: \n")

X_pred= pd.DataFrame(dataTest,columns=['sentence_count','word_count','hashtags','upper','mentions','elongated','char_count', 
'punctuation_count','commas', 'semicolons', 'exclamations', 'periods', 'questions', 'quotes', 'ellipses','title_word_count','random_cap','Links','contract_count', 
'emotion_count','number_count','retweet_count','alpha_num','sentiment','abbrev','swear'])
X_pred=preprocessing.normalize(X_pred)
X_text = data['text']
X_text_pred = dataTest['text']
# print(X.shape)
# print(y.shape)

# print(X_text)


print("*************************************")
print("Tfidf Fit: \n")
tfidf_vector = TfidfVectorizer(stop_words = 'english', ngram_range = (1,2), min_df = 1,max_features=50000)

print("Fit & Transform: \n")
X_text = tfidf_vector.fit_transform(data['text'].values.astype('U'))
print("Writing: \n")
joblib.dump(tfidf_vector,'Models/tf_idf.pkl')
X_text_tfidf_pred = tfidf_vector.transform(dataTest['text'].values.astype('U'))
joblib.dump(X_text,'Models/X_text.pkl')

# print("Shape")
# print(X_text.shape)
# print(X_text_tfidf_pred.shape)
X_text_arr = pd.DataFrame(X_text.toarray(), columns=tfidf_vector.get_feature_names())
X_text = hstack([ X_text,X])
X_text_arr_pred_arr = pd.DataFrame(X_text_tfidf_pred.toarray(), columns=tfidf_vector.get_feature_names())
X_text_tfidf_pred = hstack([X_text_tfidf_pred, X_pred])
# print("Shape after")
# print(X_text.shape)
# print(X_text_tfidf_pred.shape)
print("*************************************")

X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.05) # 70% training and 30% test

print("LinearSVC: \n")

svc = LinearSVC()
print("LinearSVC Fit: \n")
svc.fit(X_train, y_train)

print("SVC Write: ")
joblib.dump(svc, 'Models/sep4_svc.pkl')
y_pred_svc_test = svc.predict(X_test)
print("SVC Accuracy: ",metrics.accuracy_score(y_test, y_pred_svc_test))

print("Saving csv")
y_pred_svc = svc.predict(X_text_tfidf_pred)
df = pd.DataFrame({"Id":range(1, len(y_pred_svc) + 1 ,1), "Predicted":y_pred_svc})
df.to_csv('Predictions/sep6_LinearSVC_1.csv',index=False)
# Model Accuracy, how often is the classifier correct?
#print("SVC Accuracy:",metrics.accuracy_score(y_test, y_pred_svc))

print("*********************************")
