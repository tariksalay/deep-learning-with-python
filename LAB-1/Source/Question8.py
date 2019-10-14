#Read data
import pandas as pd
data = pd.read_csv('candy-data.csv')
#Handling nulls
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Features']
nulls.index.name = 'Null count'
print(nulls)
print()
corr = data.corr()
print(corr['chocolate'].sort_values(ascending=False))
# Visualize data to ind the correlation between features
import matplotlib.pyplot as plt
import seaborn as sns
sns.FacetGrid(data, hue='chocolate', height=4).map(plt.scatter, 'winpercent', 'pricepercent').add_legend()
plt.show()
sns.FacetGrid(data, hue='chocolate', height=4).map(plt.scatter, 'winpercent', 'bar').add_legend()
plt.show()
sns.FacetGrid(data, hue='chocolate', height=4).map(plt.scatter, 'fruity', 'hard').add_legend()
plt.show()
import numpy as np
data = data.select_dtypes(include=[np.number]).interpolate().dropna()
data = data.drop(['sugarpercent'], axis=1)
#Divide data into x,y
y = data['chocolate']
x = data.drop(['chocolate'], axis=1)
#Split data into test and train
from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2, random_state=0)
#Build the model
from sklearn.linear_model import LinearRegression
lrl = LinearRegression()
model = lrl.fit(x_train, y_train)
#R^2 value
print('R^2 value:', model.score(x_train, y_train))
#Mean_square_value
y_pred = model.predict(x_test)
from sklearn.metrics import mean_squared_error
print('Mean squared error:', mean_squared_error(y_test, y_pred))
#After apply EDA, the R^2 value increased 0.04, the MSE value improved 0.02