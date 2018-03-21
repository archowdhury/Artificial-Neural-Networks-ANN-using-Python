import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.width', 180)

#===============================================================================#
#    Data reading and preprocessing
#===============================================================================#

# Read in the data
#-----------------
dataset = pd.read_csv(r'C:\Users\Amit\Desktop\Python Programming\Data\Artificial_Neural_Networks\Churn_Modelling.csv')
dataset.head()
dataset.info()
dataset.shape

# Drop useless variables
#-----------------------

# The RowNumber, CustomerId and Surname are not going to be useful, so we'll drop them
dataset = dataset.drop(['RowNumber','CustomerId','Surname'], axis=1)
dataset.head()

# Separate the independent and dependent variables
#-------------------------------------------------
dataset.shape
X = dataset.iloc[:,0:10]
y = dataset.iloc[:,10]

# Create dummy variables
#-----------------------

# We'll create dummy variables for the Geography and Gender categorical variables
# drop_first creates n-1 dummy variables

X = pd.get_dummies(X, drop_first=True)
X.head()

# Split into training and test sets
#----------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=0)
len(X_train)
len(X_test)

# Feature Scaling
#----------------

# Feature scaling is very important for neural networks.
# It makes the model converge much faster, and prevents giving variables with large values
# an undue weightage

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train.head()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#===============================================================================#
#    Creating the Artificial Neural Network (ANN) model
#===============================================================================#

# Import the required libraries
#------------------------------
import keras
from keras.models import Sequential
from keras.layers import Dense   # this is used to initialize the initial weights for the ANN

# Initialize the ANN. Add the input layer, hidden layer, and the output layer
#----------------------------------------------------------------------------
classifier = Sequential()

# For this model we'll add two hidden layers with 6 neurons each
# Why 6?? Just as a simple rule of thumb to use an average of the input and output features, (11 + 1)/2 = 6
# We'll use "Rectified Linear Unit" (RELU) as the activation function
# For the final layer we'll use "Sigmoid" as the activation function. That will give us the probabilities for each of the two classes

classifier.add(Dense(units=6, input_dim=11, activation='relu'))  # the input layer, plus first hidden layer with 6 neurons
classifier.add(Dense(units=6, activation='relu'))                # the second hidden layer with 6 neurons
classifier.add(Dense(units=1, activation='sigmoid'))             # the final output layer with a "sigmoid" function as activation function

classifier.summary()                                             # just to see how many nodes have been added

# Compile the ANN
#----------------

# We'll use the 'Adam" optimiser which works well in this case
# "binary_crossentropy" has been used as the loss function since we have a binary output
# If we had more than 2 classes in the output we would have used the "categorical_crossentropy" loss function
# For the metrics we are using only the "Accuracy" for now

classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# Fit the training data to the ANN
#---------------------------------

# We are going to use "Batch Learning" with a batch size of 10 records. That will update the weights after every 10 records
# If we updated the weights after each record then it would be called "Reinforcement Learning"
# An "epoch" is one complete iteration of the entire dataset. We'll use 100 epochs, so 100 iterations over the entire dataset

classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


#===============================================================================#
#    Predicting the outputs
#===============================================================================#

# Predict the output
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Create the confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(y_test, y_pred)  # COnfusion Matrix
accuracy_score(y_test, y_pred)    # Accuracy score

# We get an accuracy of about 86% on the test set which is very good
# Hope you enjoyed this tutorial :)
