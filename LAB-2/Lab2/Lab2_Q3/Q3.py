#Question 3:  Implement the image classification with CNN model
#Dataset: https://www.kaggle.com/slothkong/10-monkey-species/data
from keras.preprocessing.image import img_to_array, load_img
import os
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD


train_path = './Lab2_Q3/training'
test_path = './Lab2_Q3/validation'

def read_data(path):
    '''This function reads images from folder'''
    images = []
    labels = []
    count = -1
    for root, folder, file in os.walk(path):
        for f in file:
            file_path = os.path.join(root, f)
            img = load_img(file_path, target_size=(32, 32))
            img = img_to_array(img)
            img = img.reshape(img.shape)
            images.append(img)
            labels.append([count])
        count += 1
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

#Read images from folders
x_train, y_train = read_data(train_path)
x_test, y_test = read_data(test_path)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0

#One hot encode data
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# #Creating model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
#
# #Compile model
# epochs = 100
# lrate = 0.01
# decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# print(model.summary())
#
# #Fitting model
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=32)
# model.save('model_Q3.h5')

#Load model and predict 4 images of the first 2 species
model = load_model('./Lab2_Q3/model_Q3.h5')
print(model.summary())

#Evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print('Loss:', score[0])
print('Accuracy: ', score[1])

print('\nFirst species:')
pred = model.predict_classes(x_test[0:4], verbose=0)
print('\tPrediction:', pred)
results = np.array(y_test[0:4])
print('\tResults:', np.argmax(results, axis=1))

print('\nSecond species:')
pred = model.predict_classes(x_test[27:31], verbose=0)
print('\tPrediction:', pred)
results = np.array(y_test[27:31])
print('\tResults:', np.argmax(results, axis=1))