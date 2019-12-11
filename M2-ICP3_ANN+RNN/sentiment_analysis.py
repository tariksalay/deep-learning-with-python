from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
import matplotlib.pyplot as plt
from keras import callbacks
df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values

#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
max_review_len = max([len(s.split()) for s in sentences])
print('Max review len:',max_review_len)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size:',vocab_size)
#getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)
#Make the same length
padded_docs = pad_sequences(sentences, maxlen=max_review_len)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
print(y)
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=0)

model = Sequential()
# BEFORE ADD EMBEDDING LAYER: loss: 0.2838 - acc: 0.9281 - val_loss: 1.0824 - val_acc: 0.5024
# AFTER:                      loss: 0.7737 - acc: 0.5829 - val_loss: 0.8513 - val_acc: 0.5089
model.add(Embedding(vocab_size, 50, input_length=2470))
model.add(Flatten())
model.add(layers.Dense(300,input_dim=2470, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Train and validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
## Unlikely

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train and validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
# Overfitting for loss (Perform well on training data, but poor on unseen data)

import numpy as np
pred = model.predict(X_test)
print(np.argmax(pred[0]))

############################################################################
# ## APPLYING FOR 20_FETCH GROUP
# #Fetching train and test data from sci-kit learn
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import TfidfVectorizer
# #Preparing train and test data
# train20 = fetch_20newsgroups(subset='train', shuffle=True)
# sentences = train20.data
# y = train20.target
#
# tokenizer = Tokenizer(num_words=2000)
# tokenizer.fit_on_texts(sentences)
# max_review_len = max([len(s.split()) for s in sentences])
# vocab_size = len(tokenizer.word_index) + 1
# sentences = tokenizer.texts_to_matrix(sentences)
# padded_docs = pad_sequences(sentences, maxlen=max_review_len)
#
# le = preprocessing.LabelEncoder()
# y = le.fit_transform(y)
# X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=0)
# model = Sequential()
#
# model.add(layers.Dense(300,input_dim=11821, activation='relu'))
# model.add(layers.Dense(20, activation='softmax'))
# model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
# history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)
# #loss: 0.2391 - acc: 0.9635 - val_loss: 0.5926 - val_acc: 0.8360
#
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Train and validation accuracy')
# plt.xlabel('Number of epochs')
# plt.ylabel('Accuracy')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show()
# ## Underfitting for accuracy (Error on the training data)
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Train and validation loss')
# plt.xlabel('Number of epochs')
# plt.ylabel('Loss')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show()
#