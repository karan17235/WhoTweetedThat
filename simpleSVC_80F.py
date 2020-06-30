import pandas as pd
import numpy as np
import string
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

stop_words = set(stopwords.words('english'))
lmt = WordNetLemmatizer()

data = pd.read_csv('../data/all_data.csv')

y=data['label']
X= pd.DataFrame(data,columns=['sentence_count','word_count','hashtags','upper','mentions','elongated','char_count',
 'punctuation_count','commas', 'semicolons', 'exclamations', 'periods', 'questions', 'quotes', 'ellipses','title_word_count','random_cap','Links','contract_count',
 'emotion_count','number_count','retweet_count','alpha_num','sentiment','abbrev','swear', 'back','best','better','blog','check','come','could','day',
 'first','free','game','get','getting','go','going',
 'good','got','great','happy','home','know','last',
 'life','like','live','ll','lol','love','make',
 'man','much','need','new','news','next','night',
 'one','people','really','right','see','show','still',
 'thank','thanks','think','time','today','tomorrow',
 'tonight','twitter','us','video','way','week','well','work','would'])
X=preprocessing.normalize(X)

data = pd.read_csv('../data/all_test.csv')


X_pred= pd.DataFrame(data,columns=['sentence_count','word_count','hashtags','upper','mentions','elongated','char_count',
 'punctuation_count','commas', 'semicolons', 'exclamations', 'periods', 'questions', 'quotes', 'ellipses','title_word_count','random_cap','Links','contract_count',
 'emotion_count','number_count','retweet_count','alpha_num','sentiment','abbrev','swear', 'back','best','better','blog','check','come','could','day',
 'first','free','game','get','getting','go','going',
 'good','got','great','happy','home','know','last',
 'life','like','live','ll','lol','love','make',
 'man','much','need','new','news','next','night',
 'one','people','really','right','see','show','still',
 'thank','thanks','think','time','today','tomorrow',
 'tonight','twitter','us','video','way','week','well','work','would'])
X_pred=preprocessing.normalize(X_pred)


print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # 70% training and 30% test
print(X_train.shape)

print("Knn")
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)
print("Knn fit")
#Train the model using the training sets
knn.fit(X_train, y_train)
print("Knn write")
joblib.dump(knn, 'ensemble/sep4_knn.pkl')
print("Knn predict")
#Predict the response for test dataset
y_pred_knn = knn.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

print("*************************************")

print("Knn")
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)
print("Knn fit")
#Train the model using the training sets
knn.fit(X, y)
print("Knn predict")
#Predict the response for test dataset
#y_pred_knn = knn.predict(X_test)

# Model Accuracy, how often is the classifier correct?
#print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Knn write")
joblib.dump(knn, 'ensemble/sep4_knn2.pkl')
y_pred_knn = knn.predict(X_pred)
df = pd.DataFrame({"Id":range(1, len(y_pred_knn) + 1 ,1), "Predicted":y_pred_knn})
df.to_csv('../predictions/sep6_knn_1.csv',index=False)

print("*************************************")

print("Svc")
svc = SVC(C=100,
                  coef0=1,
                  degree=2,
                  gamma='auto',
                  kernel='poly',
                  shrinking=False,
                  probability=True).fit(X_train, y_train)
print("Svc predict")
print("Svc write")
joblib.dump(svc, 'ensemble/sep4_svc.pkl')
y_pred_svc = svc.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("SVC Accuracy:", accuracy_score(y_test, y_pred_svc))


print("*********************************")



print("Svc")
svc = SVC(C=100,
                  coef0=1,
                  degree=2,
                  gamma='auto',
                  kernel='poly',
                  shrinking=False,
                  probability=True).fit(X, y)
print("Svc predict")
#y_pred_svc = svc.predict(X_test)
print("Svc write")
joblib.dump(svc, 'ensemble/sep4_svc2.pkl')
# Model Accuracy, how often is the classifier correct?
#print("SVC Accuracy:",metrics.accuracy_score(y_test, y_pred))


y_pred_svc = svc.predict(X_pred)
df = pd.DataFrame({"Id":range(1, len(y_pred_svc) + 1 ,1), "Predicted":y_pred_svc})
df.to_csv('../predictions/sep6_svc_1.csv',index=False)

print("************************************")




