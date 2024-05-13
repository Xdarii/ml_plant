
import pandas as pd
import sklearn 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df=pd.read_csv('crop_recommendation.csv')
print(df.head())

#lets make data ready for nmachine learning

c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes

y=df.target
X=df[['N','P','K','temperature','humidity','ph','rainfall']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)

params={"max_iter":1000, "random_state":48}
modelLR= LogisticRegression(**params)

modelLR.fit(X_train_scaled,y_train)

ypred = modelLR.predict(X_test_scaled)
acc= accuracy_score(y_pred=ypred,y_true=y_test)


print('Logistic reg Accuracy on training set: {:.2f}'.format(modelLR.score(X_train_scaled, y_train)))
print('Logistic reg Accuracy on test set: {:.2f}'.format(modelLR.score(X_test_scaled, y_test)))


clf = RandomForestClassifier(max_depth=5,n_estimators=100,random_state=42).fit(X_train_scaled, y_train)

print('RF Accuracy on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))
acc=clf.score(X_test_scaled, y_test)
print('RF Accuracy on test set: {:.2f}'.format(acc))

metrics = """
RF Accuracy on test set : {:10.4f}

![Confusion Matrix](plot.png)
""".format(acc)
with open("metrics.txt", "w") as outfile:
    outfile.write(metrics)
print("saving metrics.txt done")
# Plot it
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test_scaled, y_test, normalize="true", cmap=plt.cm.Blues)
plt.savefig("plot.png")
print("saving plot done")
