from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

#CREATE THE AUTOENCODER MODEL
# Size of encoded representations
enco_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# Input placeholder
input_img = Input(shape=(784,))
# Encode representation of the input
encoded = Dense(enco_dim, activation='relu')(input_img)
# Lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# Map input to its reconstruction
autoencoder = Model(input_img, decoded)
# Map input to its encoded representation
encoder = Model(input_img, encoded)
# Create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(enco_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
# Model compile
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

#READ DATA
(x_train, _), (x_test, _) = mnist.load_data()
# Normalize data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# Tensorboard loss graph
tb= TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)
#Fit data to the model
autoencoder.fit(x_train, x_train,
                epochs=70,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tb])

# Last epoch loss and accuracy: loss: 0.0983 - accuracy: 0.8139 - val_loss: 0.0967 - val_accuracy: 0.8130

# VISUALIZE ENCODED AND RECONSTRUCTED IMAGE
# Get the first image in test data
x= x_test[0][np.newaxis]
# Get the autoencoder prediction
prediction = autoencoder.predict(x)
# Visualize actual input
plt.imshow(x_test[0].reshape(28, 28))
plt.title("input")
plt.show()
# Visualize the prediction
plt.imshow(prediction.reshape(28, 28))
plt.title("reconstructed")
plt.show()
