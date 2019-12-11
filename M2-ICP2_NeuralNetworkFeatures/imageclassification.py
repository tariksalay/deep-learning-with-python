from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical

(train_images,train_labels),(test_images, test_labels) = mnist.load_data()


#process the data
#1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0],dimData)
test_data = test_images.reshape(test_images.shape[0],dimData)

#convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
#scale data
train_data /=255.0
test_data /=255.0
## No scalling data result: Loss = 0.15005188137479125, accuracy = 0.9580000042915344

#change the labels frominteger to one-hot encoding
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#creating network
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))

################################################################################
## PART 3: Change the number of hidden layer and the activation
model.add(Dense(300, activation='tanh'))
# Before Loss = 0.12332919966242878, accuracy = 0.9811999797821045
# After Loss = 0.10128876128743923, accuracy = 0.982699990272522
# Loss is lower

################################################################################
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

# ########################################################
## PART 1: Plot the loss and accuracy for both training data and validation data.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Train and validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
## Underfitting for accuracy (Error on the training data)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train and validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
## Overfitting for loss (Perform well on training data, but poor on unseen data)

## Training   : high            low             low         high
## Validation : high            high            low         low
## Result     : underfitting    overfitting     good-fit    unlikely

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# ################################################################################
##  PART 2: EVALUTING THE MODEL PREDICTION
pred = model.predict([test_data])
print('Prediction: ', np.argmax(pred[0]))
# Display the actual result
plt.imshow(test_images[0,:,:])
plt.show()
## The prediction was accurate