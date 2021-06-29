import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.colors import ListedColormap
import classifier
dataset = pd.read_csv('..//pythonProject3/mushrooms.csv')
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



