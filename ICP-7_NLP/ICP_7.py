# #Step 1: install wikipedia-api using 'pip install wikipedia-api'
# #Step 2: Extract the text content
# import wikipediaapi
# wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
#
# page = wiki.page('Google')
# #Write text content to a file
# fh = open('text.txt', 'w')
# fh.write(page.text)
# fh.close()

#TOKENIZATION: breaking a text stream up into list of words or sentences
import nltk

fh = open('text.txt')
text = fh.read()
fh.close()

print('WORD TOKENIZATION')
words = nltk.word_tokenize(text)
print(words)

print('SENTENCE TOKENIZATION')
sentences = nltk.sent_tokenize(text)
print(sentences)
print()

#PART-OF-SPEECH(POS): Labeling words into their part-of-speech
# CC(conjunction, coordinating)   CD(cardinal number)   DT(deteminer)
#   EX(existential there)   FW(foreign word)    IN(conjunction, subordinating, preposition)
#   JJ(adjective)   JJR(adjective, comparative)     JJS(adjective, superlative)
#   LS(list item marker)    MD(verb, modal auxillary)   NN(noun, singular, mass)
#   NNS(noun,plural)    NNP(noun, proper singular)      NNPS(noun, proper plural)

print('POS tag of words token')
pos = nltk.pos_tag(words)
print(pos)
print()

#STEMMING: reducing injected words to their stem (-ing, -ed, -s, ... to original word)

#PORTERSTEMMER: Removing suffix, simplicit, fast
#Ex: troubling => troubl
from nltk.stem import PorterStemmer
pt = PorterStemmer()
pt_list = []
for i in words:
    pt_list.append(pt.stem(i))
print('PorterStemmer result')
print(pt_list)

#LANCASTERSTEMMER: Reducing words to very original
#Ex: friendship => friend
from nltk.stem import LancasterStemmer
lc = LancasterStemmer()
lc_list = []
for i in words:
    lc_list.append(lc.stem(i))
print('Lancaster Stemmer result')
print(lc_list)

#SNOWBALLSTEMMER: Non-english stemmer
#Ex: troubling => trouble
from nltk.stem import SnowballStemmer
sb = SnowballStemmer('english')
sb_list = []
for i in words:
    sb_list.append(sb.stem(i))
print('SnowballStemmer result')
print(sb_list)
print()

#LEMMATIZATION: detemining POS and apply normalization rules for each POS
#The best result for stemming word
#Ex: better => good, cats => cat
#Related to Stemming but different: stemming doesn't consider the context meaning of a word
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
lm_list = []
for i in words:
    lm_list.append(lm.lemmatize(i))
print('Lemmatizer result')
print(lm_list)
print()

#TRIGRAM: group items as tokens of tuples
from nltk.util import ngrams
trigrams = ngrams(words, 3)
#Get the frequency
import collections
freq = collections.Counter(trigrams)
print('Trigrams result')
print(freq.most_common(10))
print()

#NAME ENTITY RECOGNITION(NER)
#Labeling categories: name of person, organizations, locations, times, quantities
from nltk import ne_chunk, pos_tag
print('Name entity recognition result')
print(ne_chunk(pos_tag(words)))
print()
###############################################################################


#TEXT CLASSIFICATION: assigning tags/categories to text base on its content
#Fetching train and test data from sci-kit learn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
#Preparing train and test data
train20 = fetch_20newsgroups(subset='train', shuffle=True)
test20 = fetch_20newsgroups(subset='test', shuffle=True)
tfidf = TfidfVectorizer()
x_train = tfidf.fit_transform(train20.data)
x_test = tfidf.transform(test20.data)

#Using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train, train20.target)
#Evaluate model
from sklearn.metrics import accuracy_score
x_pred = nb.predict(x_test)
print('Score for using Naive Bayes:', accuracy_score(test20.target, x_pred))
#score=0.77

#Using KNeighbor Classification
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(x_train, train20.target)
#Evaluate model
x_pred = kn.predict(x_test)
print('Score for using Kneighbors:', accuracy_score(test20.target, x_pred))
#n=2 => score=0.64
#n=3 => score=0.66
#n=4 => score=0.66
print()
############################################################################
#Change Tfidf to use bigram
tfidf = TfidfVectorizer(ngram_range=(1,2))
x_train = tfidf.fit_transform(train20.data)
x_test = tfidf.transform(test20.data)

#Using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train, train20.target)
#Evaluate model
from sklearn.metrics import accuracy_score
x_pred = nb.predict(x_test)
print('Applying bigram for Tfidf')
print('Score for using Naive Bayes', accuracy_score(test20.target, x_pred))
#score=0.7654
#Using KNeighbors Classification
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(x_train, train20.target)
#Evaluate model
x_pred = kn.predict(x_test)
print('Score for using KNeighbors:', accuracy_score(test20.target, x_pred))
#score=0.6131
#######################################################################
#Using stop-word 'english'
tfidf = TfidfVectorizer(stop_words='english')
x_train = tfidf.fit_transform(train20.data)
x_test = tfidf.transform(test20.data)

#Using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train, train20.target)
#Evaluate model
from sklearn.metrics import accuracy_score
x_pred = nb.predict(x_test)
print('Applying stop-word')
print('Score for using Naive Bayes:', accuracy_score(test20.target, x_pred))
#score=0.8169
#Using Kneighbors Classification
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(x_train, train20.target)
#Evaluate model
x_pred = kn.predict(x_test)
print('Score for using KNeighbors:', accuracy_score(test20.target, x_pred))
#score=0.6666
