# Artificial Neural Network 

#Install TensorFlow - theano- Keras

#TensorFlow and Theano requre building network form skratch (Used in research amd development)

#Import Keras 

# Part-1: Data preprocessing

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
# only choose columns that have impact to staying or leave bank
y = dataset.iloc[:,13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#encode categorical variables
#last line indicate creating dummy variables
#no hierarchial rank between categorical variables (country - sex)
X = X [:,1:]
#remove one dummy variable and keep 2 of other country to avoid dummy variable trap


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# change cross_validation into model_selection


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Part-2: Make ANN
#Importing the keras libraries and packages
import keras
from keras.models import Sequential #initial neural network
from keras.layers import Dense #build layers into the ANN
#We need 2 models one for initiation (Sequential) and other to add layers(Dense)

#Initializing the ANN
#2 ways to initialize the neural network (sequence of layer or graph)
#sequence of layer (classifier)
classifier = Sequential () #Initializing the ANN classifier

#Adding the input layer and first hidden layser
classifier.add (Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#hidden layer is the average between input and output layer
#rectified activation fun: hidden layer -- sigmoid activation function: output layer
#output dim: No. of nodes in hidden layer
#input_dim: imput layer no. of independant variables
# init: weights (normal distribution)

#Adding second hidden layer
classifier.add (Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#remove input dim because network is already initialized in the previous line of  codo

#Adding the output layer
classifier.add (Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#sigmoid fun. 2 categories and softmax when more than 2 categories

#Compiling the ANN
#Applying stochastic gradient descent to whole ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer: fun optimizes the weights

#Fitting the ANN to the training set
classifier.fit (X_train, y_train, batch_size = 10, nb_epoch = 100)
#batch size (Not RL): no of independant variables after which update the weight
#nb_epoch: redo after training all the training set


#Part-3: Making the predictions and evaluating the models

# Predicting the Test set results
y_pred = classifier.predict(X_test) #the result is probability
y_pred = (y_pred>0.5) #convert the vector to false and true


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
#Accuracy = (right prediction / Total prediction)