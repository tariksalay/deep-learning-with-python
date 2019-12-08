
from keras.preprocessing import sequence, text
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, Conv1D, Flatten, MaxPooling1D
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.utils.np_utils import to_categorical
import csv


#Set parameters
vocab_size = 1000
maxlen = 1000
batch_size = 32
embedding_dims = 50
filters = 16
kernel_size = 3
hidden_dims = 250
epochs = 10

#Read data
train = pd.read_csv('./train.csv', delimiter='\t')
train = train[['Phrase', 'Sentiment']]
train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
train['Phrase'] = train['Phrase'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))
for index, row in train.iterrows():
    row[0] = row[0].replace('rt', ' ')

x = train['Phrase'].values
y = train['Sentiment'].values

#Tokenize data
tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x)
x = tokenizer.texts_to_matrix(x)
x = sequence.pad_sequences(x, maxlen=maxlen)

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

print(x.shape)
print(y.shape)

#Splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Creating model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dims, input_length=maxlen))
model.add(Dropout(0.5))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_data=(x_test, y_test))
model.save('model_Q4.h5')

#Load model
model = load_model('model_Q4.h5')
print(model.summary())
score = model.evaluate(x_test, y_test, verbose=0)
print('Loss:', score[0])
print('Accuracy: ', score[1])

#Read test data
test = pd.read_csv('test.csv', delimiter='\t')
id = test['PhraseId'].values

test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())
test['Phrase'] = test['Phrase'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))
for index, row in test.iterrows():
    row[2] = row[2].replace('rt', ' ')

pred_phrases = test['Phrase'].values
#Tokenize data
tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(pred_phrases)
pred_phrases = tokenizer.texts_to_matrix(pred_phrases)
pred_phrases = sequence.pad_sequences(pred_phrases, maxlen=maxlen)

#Predict the data
pred_setiment = model.predict(pred_phrases, batch_size=batch_size, verbose=2)

#Write prediction to a new file
with open('predict.csv', mode='w') as pred_file:
    pred_writer = csv.writer(pred_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    pred_writer.writerow(['PhraseId', 'Sentiment'])
    for i in range(len(pred_setiment)):
        pred = np.argmax(pred_setiment[i])
        pred_writer.writerow([id[i], pred])

