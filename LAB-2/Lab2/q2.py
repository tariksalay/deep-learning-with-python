import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.optimizers import SGD
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.utils import np_utils
import pickle

# load dataset
dataset = pd.read_csv("heart.csv")
# dataset.info()
# print(dataset.describe())

# normalize dataset columns to 0.0-1.0
dataset = dataset.astype('float32')
dataset['age'] = dataset['age'] / max(dataset['age'])
dataset['cp'] = dataset['cp'] / max(dataset['cp'])
dataset['trestbps'] = dataset['trestbps'] / max(dataset['trestbps'])
dataset['chol'] = dataset['chol'] / max(dataset['chol'])
dataset['fbs'] = dataset['fbs'] / max(dataset['fbs'])
dataset['restecg'] = dataset['restecg'] / max(dataset['restecg'])
dataset['thalach'] = dataset['thalach'] / max(dataset['thalach'])
dataset['exang'] = dataset['exang'] / max(dataset['exang'])
dataset['oldpeak'] = dataset['oldpeak'] / max(dataset['oldpeak'])
dataset['slope'] = dataset['slope'] / max(dataset['slope'])
dataset['ca'] = dataset['ca'] / max(dataset['ca'])
dataset['thal'] = dataset['thal'] / max(dataset['thal'])

# print(dataset.describe())
# dataset.info()

all_data = dataset.drop(['target'], axis=1)
all_target = dataset['target']


X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_target,
                                                    test_size=0.30, random_state=87)

np.random.seed(123)
model = Sequential()  # create model
model.add(Dense(20, input_dim=13, activation='relu'))  # hidden layer
model.add(Dense(10, activation='relu'))  # hidden layer
model.add(Dense(1, activation='sigmoid'))  # output layer

# Model 1
epochs = 100
learning_rate = 0.005
decay_rate = learning_rate / epochs
momentum = 0.8
adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Model 2
# epochs = 150
# learning_rate = 0.001
# decay_rate = learning_rate / epochs
# momentum = 0.8
# sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)


model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])


tensorboard = TensorBoard(log_dir='./q_2_logs', histogram_freq=0,
                          write_graph=True, write_images=False)

hist = model.fit(X_train, Y_train,
                 epochs=epochs,
                 batch_size=2,
                 validation_data=(X_test, Y_test),callbacks=[tensorboard])


score = model.evaluate(X_test, Y_test)
print("test accuracy", score[1])

# loss history
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# acc
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()