import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


X, y = datasets.load_iris(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

# Define the model hyperparameters
params = {"solver": "lbfgs", "max_iter": 100, "multi_class": "auto", "random_state": 8888}

modelLR=LogisticRegression(**params)

# build the model
modelLR.fit(X_train,y_train)


# Predict on the test set
y_pred = modelLR.predict(X_test)

# Calculate accuracy as a target loss metric
accuracy = accuracy_score(y_test, y_pred)

print("Test accuracy:",accuracy)

