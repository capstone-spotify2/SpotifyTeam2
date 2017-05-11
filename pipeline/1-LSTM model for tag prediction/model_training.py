from keras.utils import np_utils
import re
import pandas as pd
import nltk.data
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
import gensim
from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models import Phrases
import logging
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.preprocessing import LabelEncoder

def sentence_to_wordlist(sentence, remove_stopwords=False):
    # 1. Remove non-letters
    sentence_text = re.sub(r'[^\w\s]','', sentence)
    # 2. Convert words to lower case and split them
    words = sentence_text.lower().split()
    if remove_stopwords:
        stop = set(stopwords.words('english'))
        filter(lambda x: x not in stop, stop)
    # 3. Return a list of words
    return(words)


##############################
#
# word2vec + LSTM
#
#####

#from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.engine import Input
from keras.preprocessing import sequence

print("---- Load in Trained Word2vec Model ----")
## load pre-trained word2vec dataset
from gensim.models.keyedvectors import KeyedVectors
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


weights = word2vec.syn0
layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
vocab = dict([(k, v.index) for k, v in word2vec.vocab.items()])
json = json.dumps(vocab)
f = open("vocab.json","w")
f.write(json)
f.close()

print("---- Load in Lyrics Database ----")
## load matched song lyrics with tags
tagdf = pd.read_csv('full_dataset.csv', index_col = 0)
newind = np.arange(tagdf.shape[0])
## randomly shuffle
np.random.shuffle(newind)

print("---- Train/Test Set Split ----")
## split out training, testing, and validation set
train_df = tagdf.ix[newind[:int(tagdf.shape[0]*0.9*0.7)]]
test_df = tagdf.ix[newind[int(tagdf.shape[0]*0.9*0.7):int(tagdf.shape[0]*0.9)]]
validate_df = tagdf.ix[newind[int(tagdf.shape[0]*0.9):]]

print("---- Convert Raw Lyrics to Word Vectors ----")
x_train = []
i = 0
for lyrics in train_df.Song_lyrics:
    #import pdb;pdb.set_trace()
    #print('hahah')
    wordlist = sentence_to_wordlist(lyrics)
    vecinput = [vocab[w] for w in wordlist if w in vocab]
    x_train.append(vecinput)
    i += 1
    if i%100 == 0:
        print i,


x_test = []
i = 0
for lyrics in test_df.Song_lyrics:
    wordlist = sentence_to_wordlist(lyrics)
    vecinput = [vocab[w] for w in wordlist if w in vocab]
    x_test.append(vecinput)
    i += 1
    if i%100 == 0:
        print i,


x_all = []
i = 0
for lyrics in tagdf.Song_lyrics:
    wordlist = sentence_to_wordlist(lyrics)
    vecinput = [vocab[w] for w in wordlist if w in vocab]
    x_all.append(vecinput)
    i += 1
    if i%100 == 0:
        print i,


x_train = np.array(x_train).reshape((train_df.shape[0],))
x_test = np.array(x_test).reshape((test_df.shape[0],))
x_train2 =sequence.pad_sequences(x_train, 150)
x_test2 =sequence.pad_sequences(x_test, 150)

x_all = np.array(x_all).reshape((tagdf.shape[0],))
x_all2 =sequence.pad_sequences(x_all, 150)



alltags = tagdf.tag.unique()

## for binary model

# tag = 'happy'
# tagmap = {alltags[i]: i for i in range(len(alltags))}
# train_df['tag_num'] = train_df.tag.map(lambda x: tagmap[x])
# test_df['tag_num'] = test_df.tag.map(lambda x: tagmap[x])
# model = Sequential()
# model.add(layer)
# model.add(LSTM(100))
# model.add(Dense(len(alltags), activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train2, np.array(train_df.tag).reshape(train_df.shape[0],), validation_data=(x_test2, np.array(test_df.tag).reshape(test_df.shape[0],)), epochs = 3)
# model.fit(x_train2, y_train, validation_data=(x_test2, y_test), epochs = 3)
# model.save('happy_model.h5')
# pred = model.predict(x_test2)
# np.sum(pred.T>0.5)

print("---- Convert y Values ----")
## for multiclass model
encoder = LabelEncoder()
encoder.fit(train_df.tag)
encoded_Y = encoder.transform(train_df.tag)
# convert integers to dummy variables (i.e. one hot encoded)
y_train = np_utils.to_categorical(encoded_Y)

encoder = LabelEncoder()
encoder.fit(test_df.tag)
encoded_Y = encoder.transform(test_df.tag)
# convert integers to dummy variables (i.e. one hot encoded)
y_test = np_utils.to_categorical(encoded_Y)

encoder = LabelEncoder()
encoder.fit(tagdf.tag)
encoded_Y = encoder.transform(tagdf.tag)
# convert integers to dummy variables (i.e. one hot encoded)
y_all = np_utils.to_categorical(encoded_Y)

print("---- Start Training the LSTM Model ----")
model = Sequential()
model.add(layer)
model.add(LSTM(100))
model.add(Dense(len(alltags), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train2, y_train, validation_data=(x_test2, y_test), epochs = 3)
# model.save('multiclass_model.h5')

print("---- Done Model Training ----")

