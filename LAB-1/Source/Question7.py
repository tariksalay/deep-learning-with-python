#Read data from file
fh = open('nlp_input.txt')
text = fh.read()
fh.close()

import nltk
#Tokenization
words = nltk.word_tokenize(text)
sentences = nltk.sent_tokenize(text)
#Apply lemmatization
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
lem_lst = []
for i in words:
    lem_lst.append(lm.lemmatize(i))
print('Tokenization result')
print(lem_lst)
print()

#Remove all punctuations in word token
new_words = []
for i in words:
    if i not in ',.?=;:!()':
        new_words.append(i)
#Find trigrams
from nltk.util import ngrams
trigrams = ngrams(new_words, 3)
#Get the frequency of trigrams
import collections
freq = collections.Counter(trigrams)
print('Trigrams result')
#Top 10 most frequency trigrams
most_freq = freq.most_common(10)
print(most_freq)
print()

#Find sentences contain most frequence trigrams
lst = []
for i in most_freq:
    #Join the trigrams tokens
    str = ' '.join(i[0])
    lst.append(str)
trigrams_sentences = ''
for i in sentences:
    for j in lst:
        if j in i:
            #Get the sentence contain the trigrams
            trigrams_sentences += i
            break
print('Sentences contain most frequency trigrams')
print(trigrams_sentences)