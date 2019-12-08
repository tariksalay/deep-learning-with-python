import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Set parameters
max_fatures = 1000
batch_size = 60
embed_dim = 50
lstm_out = 196

#Read train data
data = pd.read_csv('train.csv', delimiter='\t')
# Keeping only the neccessary columns
data = data[['Phrase','Sentiment']]
data['Phrase'] = data['Phrase'].apply(lambda x: x.lower())
data['Phrase'] = data['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

X = data['Phrase'].values
y = data['Sentiment'].values

#Tokenize data
tokenizer = Tokenizer(num_words=max_fatures)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=max_fatures)

#Encoding sentiment
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(y)
y = to_categorical(integer_encoded)
print(X.shape)
print(y.shape)

#Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model
# print(model.summary())

model = createmodel()
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2, validation_data=(X_test, Y_test))

#Save the model
model.save('model_Q4.h5')

# #Load model
# model = load_model('model_Q5.h5')
score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)
print(model.summary())
print('Loss:',score)
print('Accuracy:',acc)
#
# #Read test data
# test = pd.read_csv('test.csv', delimiter='\t')
# id = test['PhraseId'].values
#
# test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())
# test['Phrase'] = test['Phrase'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))
# for index, row in test.iterrows():
#     row[2] = row[2].replace('rt', ' ')
#
# pred_phrases = test['Phrase'].values
# #Tokenize data
# tokenizer = Tokenizer(num_words=max_fatures)
# tokenizer.fit_on_texts(pred_phrases)
# pred_phrases = tokenizer.texts_to_sequences(pred_phrases)
# pred_phrases = pad_sequences(pred_phrases, maxlen=max_fatures)
# print(pred_phrases.shape)
# #Predict the data
# pred_setiment = model.predict(pred_phrases, batch_size=batch_size, verbose=2)
#
# #Write prediction to a new file
# with open('predict.csv', mode='w') as pred_file:
#     pred_writer = csv.writer(pred_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     pred_writer.writerow(['PhraseId', 'Sentiment'])
#     for i in range(len(pred_setiment)):
#         pred = np.argmax(pred_setiment[i])
#         pred_writer.writerow([id[i], pred])
#
