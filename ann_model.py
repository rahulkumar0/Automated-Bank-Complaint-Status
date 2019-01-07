import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y = train['Complaint-Status']
train = train.drop(['Complaint-Status'],axis =1)
total = pd.concat([train,test],ignore_index = True)
summary = total['Consumer-complaint-summary']
total = total.drop(['Company-response','Consumer-disputes','Complaint-ID','Consumer-complaint-summary'],axis=1)

#check null values
total1 = total.isnull().sum().sort_values(ascending=False)
percent = (total.isnull().sum()/total.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total1, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#processing date datatype
date_format = "%m/%d/%Y"
days = []
for i in range(0,61809):
    a = datetime.strptime(total.loc[i,'Date-received'],date_format)
    b = datetime.strptime(total.loc[i,'Date-sent-to-company'], date_format)
    delta = b-a
    days.append(delta.days)
    
days = np.array(days)
days = pd.Series(days,name = 'respond days')
total = pd.concat([total,days],axis = 1)
total = total.drop(['Date-received','Date-sent-to-company'],axis = 1)
total.nunique()

#encoding
total = pd.get_dummies(total)
total = total.drop(['Transaction-Type_Bank account or service'],axis = 1)
y= y.astype('category')
labelencoder_X = LabelEncoder()
y= labelencoder_X.fit_transform(y)
y = pd.get_dummies(y)
"""
# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 61809):
    review = re.sub('[^a-zA-Z]', ' ', summary[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english')
                                                                    +stopwords.words('spanish')
                                                                    +stopwords.words('french'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
#saving list
import pickle
with open("corpus.txt", "wb") as fp:
    pickle.dump(corpus, fp)
"""
#load list
import pickle
with open("corpus.txt", "rb") as fp:
    corpus = pickle.load(fp)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
summary = cv.fit_transform(corpus).toarray()
total = np.concatenate((total.values,summary),axis = 1)
#divide train test
X = total[0:43266,:]
Xtest = total[43266:,:]

#applying xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
model = XGBClassifier(n_estimators=90,learning_rate=0.2,max_depth=7,objective = 'multi:softmax',
                       num_class=5,n_jobs=-1)
model.fit(X,y)
plot_importance(model)
plt.show()
thresholds = np.sort(model.feature_importances_)
# select features using threshold
selection = SelectFromModel(model, threshold=0.0004, prefit=True)
select_X_train = selection.transform(X)
select_X_test = selection.transform(Xtest)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 800, init = 'uniform', activation = 'relu', input_dim = 1597))
classifier.add(Dropout(0.1))

#adding second hideen layer
classifier.add(Dense(output_dim = 255, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.1))

# Adding the output layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(select_X_train, y, batch_size = 40,nb_epoch = 10)



# train model
selection_model = XGBClassifier(n_estimators=100,learning_rate=0.2,max_depth=7,objective = 'multi:softmax',
                       num_class=5,n_jobs=-1)
selection_model.fit(select_X_train, y)
# eval model

y_pred = selection_model.predict(select_X_test)

y_pred = model.predict(X_test)
print(np.sum( y_pred != y_test) / float(y_test.shape[0]))

#applying  xg
model1 = XGBClassifier(n_estimators=80,learning_rate=0.3,max_depth=7,objective = 'multi:softmax',
                       num_class=5,n_jobs=-1)
model1.fit(select_X_train,y)
y_pred = model1.predict(select_X_test)

y_pred = labelencoder_X.inverse_transform(y_pred)
y_pred = y_pred.astype(str)

y_pred = pd.Series(y_pred,name="Complaint-Status")
submission = pd.concat([test['Complaint-ID'],y_pred],axis=1)
submission.to_csv("submission45.csv",index=False)

#applying  ann

y_pred = classifier.predict(select_X_test)
y_pred = np.argmax(y_pred,axis = 1)

y_pred = labelencoder_X.inverse_transform(y_pred)
y_pred = y_pred.astype(str)

y_pred = pd.Series(y_pred,name="Complaint-Status")
submission = pd.concat([test['Complaint-ID'],y_pred],axis=1)
submission.to_csv("submission38.csv",index=False)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()