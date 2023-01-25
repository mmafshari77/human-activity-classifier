import pandas as pd
X_train = pd.read_fwf("train/X_train.txt", header=None)
y_train = pd.read_fwf("train/y_train.txt", header=None)
X_test = pd.read_fwf("test/X_test.txt", header=None)
y_test = pd.read_fwf("test/y_test.txt", header=None)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import classification_report
dctr = tree.DecisionTreeClassifier()
dctr.fit(X_train, y_train)
model = KNeighborsClassifier()
model.fit(X_train,y_train)

print()
print("This is the result using K-Neighbors algorithm:")
print(classification_report(model.predict(X_test), y_test))

print()
print("##################")
print()
print("This is the result using Decision Tree algorithm:")
print(classification_report(dctr.predict(X_test), y_test))