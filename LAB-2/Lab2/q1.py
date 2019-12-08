import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
from keras import metrics
import pickle

# load dataset
dataset = pd.read_csv("diamonds.csv")
# print(dataset)
# dataset.info()  # No null value

# Wrangling the non-numeric Features
# cut           53940 non-null object
# color         53940 non-null object
# clarity       53940 non-null object

le = LabelEncoder()
dataset['enc_cut'] = le.fit_transform(dataset['cut'].astype('str'))
dataset['enc_color'] = le.fit_transform(dataset['color'].astype('str'))
dataset['enc_clarity'] = le.fit_transform(dataset['clarity'].astype('str'))

all_data = dataset.drop(['Unnamed: 0', 'price', 'cut', 'color', 'clarity'], axis=1)
# all_train_data.info()
all_price = dataset['price']

X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_price,
                                                    test_size=0.30, random_state=87)
np.random.seed(123)

# model 1
epochs = 20
learning_rate = 0.005
batch_size = 64
decay_rate = learning_rate / epochs
activation_function = 'linear'
adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# model 2
# epochs = 20
# learning_rate = 0.001
# batch_size = 32
# optimizer = 'Adamax'
# activation_function = 'linear'

model = Sequential()  # create model
model.add(Dense(50, input_dim=9, activation=activation_function))  # hidden layer
model.add(Dropout(0.1))
model.add(Dense(20, activation=activation_function))
model.add(Dense(10, activation=activation_function))
# my_first_nn.add(Dense(10, activation='sigmoid')) # output layer
model.add(Dense(1, input_dim=9, activation="relu"))  # output layer
model.compile(loss='mean_squared_error', optimizer=adam, metrics=[metrics.mae])


# tensorboard --logdir=./
tensorboard = TensorBoard(log_dir='./q_1_logs', histogram_freq=0,
                          write_graph=True, write_images=False)

hist = model.fit(X_train, Y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_test, Y_test),
                 callbacks=[tensorboard])

mae, loss = model.evaluate(X_test, Y_test, verbose=0)
print("The mae is: ", mae)
print("The loss is: ", loss)

# accuracy history
plt.plot(hist.history['mean_absolute_error'])
plt.plot(hist.history['val_mean_absolute_error'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()