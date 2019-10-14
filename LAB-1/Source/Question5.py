import warnings
warnings.filterwarnings("ignore")
import pandas as pd
data = pd.read_csv('cars.csv', delimiter=',', header=None, skiprows=1, names=['mpg','cylinders','cubicinches','hp','weightlbs','time-to-60','year','brand'])

#Count the number of classes in the target 'brand'
print(data['brand'].value_counts(dropna=False))

#Nulls Handling
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Features']
nulls.index.name = 'Nulls count'
print(nulls)
#No nulls were found
#Visualize data to analyze our feature correlations
import seaborn as sns
import matplotlib.pyplot as plt
sns.FacetGrid(data, hue='brand', height=4).map(plt.scatter, 'mpg', 'cylinders').add_legend()
plt.show()
sns.FacetGrid(data, hue='brand', height=4).map(plt.scatter, 'cubicinches', 'hp').add_legend()
plt.show()
sns.FacetGrid(data, hue='brand', height=4).map(plt.scatter, 'weightlbs', 'time-to-60').add_legend()
plt.show()

#Encoding non-numeric features
from sklearn.preprocessing import LabelEncoder
data = data.apply(LabelEncoder().fit_transform)
#Split data into train and test
from sklearn.model_selection import train_test_split
x = data.drop(['brand'], axis=1)
y = data['brand']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=0)

#Naive Bayes method
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
nb = GaussianNB()
nb.fit(x_train, y_train)
#Evaluate model
#.score() for train data calculate the different between y_train from model and accuracy measure y_train
score = nb.score(x_train, y_train)
print('Naive Bayes accuracy training score: ', score)
print('Classification report:')
y_pred = nb.predict(x_test)
print(classification_report(y_test, y_pred))
print()

#KNN method
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
#Evaluate model
score = knn.score(x_train, y_train)
print('K-Neighbors accuracy training score: ', score)
print('Classification report:')
y_pred = knn.predict(x_test)
print(classification_report(y_test, y_pred))
print()

#SVM method
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
#Evalueate model
score = svc.score(x_train, y_train)
print('Support Vector Machines score: ', score)
print('Classification report:')
y_pred = svc.predict(x_test)
print(classification_report(y_test, y_pred))

#Support Vector Machines algorithm performs with the best result
