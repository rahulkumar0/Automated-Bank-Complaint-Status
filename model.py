import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_test = pd.reaf_csv('y_test.csv')
y = train['Complaint-Status']
train = train.drop(['Complaint-Status'],axis =1)
total = pd.concat([train,test],ignore_index = True)
summary = total['Consumer-complaint-summary'] #For nlp processing
total = total.drop(['Complaint-ID','Consumer-complaint-summary'],axis=1)


#check null values
total1 = total.isnull().sum().sort_values(ascending=False)
percent = (total.isnull().sum()/total.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total1, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
total = total.drop(['Company-response','Consumer-disputes'],axis=1) #since there are 16% and 21% missing values of both features.

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
labelencoder_y = LabelEncoder()
y_test= labelencoder_y.fit_transform(y_test)
y = pd.get_dummies(y) #for ann model

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(total)):
    review = re.sub('[^a-zA-Z]', ' ', summary[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english')
                                                                    +stopwords.words('spanish')
                                                                    +stopwords.words('french'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
summary = cv.fit_transform(corpus).toarray()
total = np.concatenate((total.values,summary),axis = 1)
#divide train test
X = total[0:len(train),:]
Xtest = total[len(train):,:]

#selecting important features using xgboost
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
selection = SelectFromModel(model, threshold=0.0005, prefit=True)
select_X_train = selection.transform(X)
select_X_test = selection.transform(Xtest)
#creating validation dataset
from sklearn.model_selection import train_test_split
X_test, X_val, y_test, y_val = train_test_split(select_X_test, y_test, test_size = 0.1, random_state = 0)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# ANN Model
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 255, init = 'uniform', activation = 'relu', input_dim = 531))
classifier.add(Dropout(0.2))

#adding second hideen layer
classifier.add(Dense(output_dim = 255, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(select_X_train, y, batch_size = 40,validation_data=(X_val,y_val),nb_epoch = 30)

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

#validation functions
def validation(model):
    yprob = model.predict( X_test )
    return np.sum( yprob != y_test) / float(y_test.shape[0])

def multiclass_logloss(actual, predicted, eps=1e-15):
    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

#Evaluating model
print(validation(classifier))
print(multiclass_logloss(y_test,classifier.predict(X_test)))


#Applying xgboost
model = XGBClassifier(n_estimators=100,learning_rate=0.2,max_depth=7,objective = 'multi:softmax',
                       num_class=5,n_jobs=-1)
#training
eval_set = [(X_val, y_val)]
model.fit(select_X_train, y,eval_set=eval_set,eval_metric='mlogloss',early_stopping_rounds=10,verbose = 1)

# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()
#Evaluating model
print(validation(model))
print(multiclass_logloss(y_test,model.predict(X_test)))

#Parameter tuning for xgboost model
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'max_depth': [4, 5, 6], 'learning_rate': [0.1,0.2,0.3],'min_child_weight':[1,2,3,4,5]}]
              
grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(selection_X_train, y)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
