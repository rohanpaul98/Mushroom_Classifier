import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.colors import ListedColormap
import classifier
dataset = pd.read_csv('../Mushroom_Classification/mushrooms.csv')
# print(dataset.head())
# print(dataset.isnull().sum())
# print(dataset.head().describe())
x = dataset.drop('class',axis=1)
y = dataset['class']
# print(y)
Encoder_x = LabelEncoder()

for col in x.columns:
    x[col] = Encoder_x.fit_transform(x[col])

Encoder_y = LabelEncoder()
y = Encoder_y.fit_transform(y)
# print(y)

x = pd.get_dummies(x,columns=x.columns,drop_first=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
# print(x_test)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# print(x_test)

pca = PCA(n_components=2)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# print(x_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(8, kernel_initializer='uniform', activation= 'relu', input_dim = 2))
classifier.add(Dense(6, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dense(5, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dense(4, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dense(1, kernel_initializer= 'uniform', activation= 'sigmoid'))
classifier.compile(optimizer= 'adam',loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train,y_train,batch_size = 10,epochs=100)


y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)



print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))

# visualization_train(model='ANN')
# visualization_test(model='ANN')

def print_score(classifier,X_train,y_train,X_test,y_test,train=True):
    if train == True:
        print("Training results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train,classifier.predict(X_train))))
        print('Classification Report:\n{}\n'.format(classification_report(y_train,classifier.predict(X_train))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_train,classifier.predict(X_train))))
        res = cross_val_score(classifier, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')
        print('Average Accuracy:\t{0:.4f}\n'.format(res.mean()))
        print('Standard Deviation:\t{0:.4f}'.format(res.std()))
    elif train == False:
        print("Test results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_test,classifier.predict(X_test))))
        print('Classification Report:\n{}\n'.format(classification_report(y_test,classifier.predict(X_test))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_test,classifier.predict(X_test))))

classifier = LogisticRegression()
classifier.fit(x_train,y_train)
print_score(classifier,x_train,y_train,x_test,y_test,train=True)
print_score(classifier,x_train,y_train,x_test,y_test,train=False)
# visualization_train('Logistic Reg')
# visualization_test('Logistic Reg')

classifier = SVC(kernel='rbf',random_state=42)
classifier.fit(x_train,y_train)

print_score(classifier,x_train,y_train,x_test,y_test,train=True)
print_score(classifier,x_train,y_train,x_test,y_test,train=False)

# visualization_train('SVC')
# visualization_test('SVC')

classifier = KNN()
classifier.fit(x_train,y_train)

print_score(classifier,x_train,y_train,x_test,y_test,train=True)
print_score(classifier,x_train,y_train,x_test,y_test,train=False)

# visualization_train('K-NN')
# visualization_test('K-NN')

classifier = NB()
classifier.fit(x_train,y_train)

print_score(classifier,x_train,y_train,x_test,y_test,train=True)
print_score(classifier,x_train,y_train,x_test,y_test,train=False)
# visualization_train('Naive Bayes')
# visualization_test('Naive Bayes')

classifier = DT(criterion='entropy',random_state=42)
classifier.fit(x_train,y_train)

print_score(classifier,x_train,y_train,x_test,y_test,train=True)
print_score(classifier,x_train,y_train,x_test,y_test,train=False)

classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42)
classifier.fit(x_train, y_train)

print_score(classifier,x_train,y_train,x_test,y_test,train=True)
print_score(classifier,x_train,y_train,x_test,y_test,train=False)
