import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import seaborn as sns

#Set up the output screen
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = [10, 6]
#Read data
train1 = pd.read_csv('./train.csv')
#Display the scatter plot of GarageArea and SalePrice
plt.scatter(train1.GarageArea, train1.SalePrice, color='blue')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()
#Delete the outlier value of GarageArea
outlier_drop = train1[(train1.GarageArea < 1000) & (train1.GarageArea > 180)]
##Display the scatter plot of GarageArea and SalePrice after deleting
plt.scatter(outlier_drop.GarageArea, outlier_drop.SalePrice, color='purple')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()


#PART 2
#Read data
train2 = pd.read_csv('./winequality-red.csv')
#Get the top 3 most correlated features
corr = train2.corr()

plt.imshow(corr, cmap="YlGnBu")
plt.colorbar()
plt.xticks(range(len(corr)),corr.columns, rotation=20)
plt.yticks(range(len(corr)),corr.index)
plt.show()
#Print out total nulls
nulls = pd.DataFrame(train2.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Features'
print(nulls)
#Delete the null values
data = train2.select_dtypes(include=[np.number]).interpolate().dropna()
print()
print('Now, total nulls in data is: ', sum(data.isnull().sum() != 0))
#Buils the linear model
y = np.log(train2.quality)
x = data.drop(['quality'], axis=1)
#Split data into test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.33)
#Create the model
lrl = linear_model.LinearRegression()
model = lrl.fit(x_train, y_train)
#R2 value
print('R^2 value is:',model.score(x_test, y_test))
#RMSE value
pred = model.predict(x_test)
print('RMSE value is: ', mean_squared_error(y_test, pred))