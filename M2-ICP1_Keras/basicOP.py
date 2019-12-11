from keras.models import Sequential
from keras.layers.core import Dense
#Read data
import pandas as pd
data = pd.read_csv('diabetes.csv', header=None).values
print(data)
#Split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[:,:8], data[:,8], test_size=0.25, random_state=0)
#Apply model
import numpy as np
np.random.seed(155)
nn = Sequential()
#Add the first layer with 8 features(dimentions) and 20 neurals
nn.add(Dense(20, input_dim=8, activation='relu'))

#Add another layer that have 10 neurals
#Score descrease 0.69=>0.61
nn.add(Dense(10, activation='relu'))
nn.add(Dense(5, activation='relu'))

#Last layer with 1 neural
nn.add(Dense(1, activation='sigmoid'))
#Configure the learning process
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_fitted = nn.fit(x_train, y_train, epochs=100, verbose=0, initial_epoch=0)
print(nn.summary())
#Evaluate model
print(nn.evaluate(x_test, y_test, verbose=0))


#Part 2
#Read data
data2 = pd.read_csv('breastcancer.csv')
#Encoding
data2['diagnosis'] = data2['diagnosis'].replace('M', 0)
data2['diagnosis'] = data2['diagnosis'].replace('B', 1)
x = data2.iloc[:,2:32]
y = data2.iloc[:,1]
#Spliting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
#Apply model
nn2 = Sequential()
nn2.add(Dense(40, input_dim=30, activation='relu'))
nn2.add(Dense(25, activation='relu'))
nn2.add(Dense(8, activation='tanh'))
nn2.add(Dense(1, activation='sigmoid'))
nn2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn2_fitted = nn2.fit(x_train, y_train, epochs=100, verbose=0, initial_epoch=0)
#Evaluate
print(nn2.summary())
print(nn2.evaluate(x_test, y_test, verbose=0))

#Part 3
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
#Normalization
ss.fit(x)
x_scaled = pd.DataFrame(ss.transform(x), columns=x.columns)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=0)
#Apply model
nn3 = Sequential()
nn3.add(Dense(40, input_dim=30, activation='relu'))
nn3.add(Dense(25, activation='relu'))
nn3.add(Dense(8, activation='tanh'))
nn3.add(Dense(1, activation='sigmoid'))
nn3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn3_fitted = nn3.fit(x_train, y_train, epochs=100, verbose=0, initial_epoch=0)
print(nn3.summary())
print(nn3.evaluate(x_test, y_test, verbose=0))