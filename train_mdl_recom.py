
import pandas as pd
import sklearn 
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import xgboost as xgb

df=pd.read_csv('crop_recommendation.csv')
print(df.head())

#lets make data ready for nmachine learning

c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes

y=df.target
X=df[['N','P','K','temperature','humidity','ph','rainfall']]

nclass=len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)

params={"max_iter":1000, "random_state":48}
# =======================================================
# Logistic Regression
modelLR= LogisticRegression(**params)

modelLR.fit(X_train_scaled,y_train)

ypred = modelLR.predict(X_test_scaled)
acc= accuracy_score(y_pred=ypred,y_true=y_test)


print('Logistic reg Accuracy on training set: {:.2f}'.format(modelLR.score(X_train_scaled, y_train)))
print('Logistic reg Accuracy on test set: {:.2f}'.format(modelLR.score(X_test_scaled, y_test)))
disp = ConfusionMatrixDisplay.from_estimator(modelLR, X_test_scaled, y_test, normalize="true", cmap=plt.cm.Blues)
plt.savefig("plotLR.png")
print("saving plot done")
# =======================================================
# Random Forest
clf = RandomForestClassifier(max_depth=6,n_estimators=1000,random_state=42).fit(X_train_scaled, y_train)

print('RF Accuracy on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))
acc=clf.score(X_test_scaled, y_test)
print('RF Accuracy on test set: {:.2f}'.format(acc))

metrics = """
RF Accuracy on test set : {:10.4f}

""".format(acc)
with open("metrics.txt", "w") as outfile:
    outfile.write(metrics)
print("saving metrics.txt done")
# Plot it
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test_scaled, y_test, normalize="true", cmap=plt.cm.Blues)
plt.savefig("./plotRF.png")
print("saving plot done")


# =======================================================
# Xgboost

xgb_model=xgb.XGBClassifier(objective="multi:softmax",num_class= nclass,random_state=42)


xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)])

y_pred = xgb_model.predict(X_test)

print('Xgboost  Accuracy on training set: {:.2f}'.format(xgb_model.score(X_train_scaled, y_train)))
print('Xgboost  Accuracy on test set: {:.2f}'.format(xgb_model.score(X_test_scaled, y_test)))
disp = ConfusionMatrixDisplay.from_estimator(xgb_model, X_test_scaled, y_test, normalize="true", cmap=plt.cm.Blues)
plt.savefig("plotXgb.png")
print("saving plot done")

# Metrics calculations
acc_LR_train = modelLR.score(X_train_scaled, y_train)
acc_LR_test = modelLR.score(X_test_scaled, y_test)

acc_RF_train = clf.score(X_train_scaled, y_train)
acc_RF_test = clf.score(X_test_scaled, y_test)

acc_XGB_train = xgb_model.score(X_train_scaled, y_train)
acc_XGB_test = xgb_model.score(X_test_scaled, y_test)

# Print metrics to console and save them to file
metrics = f"""
Logistic Regression Accuracy on training set: {acc_LR_train:.2f}
Logistic Regression Accuracy on test set: {acc_LR_test:.2f}
![](./plotLR.png "Confusion Matrix LR")
Random Forest Accuracy on training set: {acc_RF_train:.2f}
Random Forest Accuracy on test set: {acc_RF_test:.2f}
![](./plotRF.png "Confusion Matrix RF")
XGBoost Accuracy on training set: {acc_XGB_train:.2f}
XGBoost Accuracy on test set: {acc_XGB_test:.2f}
![](./plotXgb.png "Confusion Matrix RF")

"""
with open("metrics.txt", "w") as outfile:
    outfile.write(metrics)
 

print("Metrics saved to 'metrics.txt'")