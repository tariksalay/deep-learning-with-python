from pathlib import Path
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv(Path(__file__).with_name('glass.csv'))
y_df = df['Type']
x_df = df.drop('Type', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=0)
nb = GaussianNB()
nb.fit(X_train, y_train)


Y_pred = nb.predict(X_test)
acc_svc = round(nb.score(X_test, y_test) * 100, 2)
print("Naive Bayes accuracy is:", acc_svc)
print("Classification report is:", classification_report(y_test, Y_pred, zero_division=0))


svm = SVC()
svm.fit(X_train, y_train)


Y_pred = svm.predict(X_test)
acc_svc = round(svm.score(X_test, y_test) * 100, 2)
print("SVM accuracy is:", acc_svc)
print("Classification report is:", classification_report(y_test, Y_pred, zero_division=0))
