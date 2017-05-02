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
## to solve the ascii decoding error
import sys
reload(sys)
sys.setdefaultencoding('utf8')

nltk.download('punkt')
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

def LyricsToSentences(lyrics):
    """
    This function achieves the following:
    - process raw lyrics to remove tags such as "chorus"
    - break the raw text into sentences
    - in order to avoid extreming short sentences that are
      often out of context, we apply a heuristic where we
      merge any lyric line without a verb with the previous line
    """
    if '\r' in lyrics:
        lines = lyrics.split('\r')
        for i, l in enumerate(lines):
            if "<i>" in l:
                lines[i] = ""
            if 'VB' not in [t[:2] for w,t in nltk.pos_tag(word_tokenize(l))]:
                lines[i] = l.replace("\n", "")
        lines = ' '.join(lines).replace("\n \n", "\n").split(' \n')
    elif "\n" in lyrics:
        lines = lyrics.split('\n')
        for i, l in enumerate(lines):
            if "<i>" in l:
                lines[i] = ""
            elif 'VB' in [t[:2] for w,t in nltk.pos_tag(word_tokenize(l))]:
                lines[i] = l + "\r"#l.replace("\n", "")
        lines = ' '.join(lines).split('\r')
        
    return lines

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

def oped_to_sentences(raw_sentences, remove_stopwords=False ):
    """
    Return the list of sentences (each sentence is a list of words, so this returns a list of lists)
    """
    try:
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                sentences.append(sentence_to_wordlist(raw_sentence, remove_stopwords))
        len(sentences)
        return sentences
    except:
        print('nope')


def mostSimilarFromList(inputw, wlist, wmodel):
    if inputw in wmodel.vocab.keys():
        sims = [wmodel.similarity(inputw, w) for w in wlist]
        return wlist[np.argmax(sims)]
    else:
        print ("input not in vocabulary")
        return None



dir = "GetAlbumInfo/lyrics/"
lfiles = [dir+f for f in listdir(dir) if isfile(join(dir, f)) and f.endswith(".txt")]

for i, f in enumerate(lfiles[486+521+4+18+7+10+11+45+293+21+144+165+24847+126+13281:]):
    print f
    file = open(f,'r') 
    lyrics = file.read()
    if (' math' in lyrics.lower())& ('learned' in lyrics.lower()):
        break

sentences = []

for f in lfiles:
    print f
    file = open(f,'r') 
    lyrics = file.read()
    try:
        raw_sentences = LyricsToSentences(lyrics)
        sentences += oped_to_sentences(raw_sentences, remove_stopwords=True)
    except:
        print 'except', f

print("There are " + str(len(sentences)) + " sentences in our corpus of opeds.")
# There are 1731157 sentences in our corpus of opeds.
np.mean([len(s) for s in sentences])
# Out[53]: 7.8848290478564333

for i, s in enumerate(sentences):
    sentences[i] = filter(lambda x: x not in stop, s)

##############################
#
# add literature
#
#####
tokenizer = RegexpTokenizer(r'\w+')
puncs = [',', '.', ']', '[', '!', '?']
lits = []
for lit in nltk.corpus.gutenberg.fileids():
    sents = list(nltk.corpus.gutenberg.sents(lit))
    for i, s in enumerate(sents):
        sents[i] = filter(lambda x: (x not in stop)&(x not in puncs), s)
    lits+=sents

lits = sentences+lits

from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)

##############################
#
# building model
#
#####

num_features = 3    # Word vector dimensionality                      
min_word_count = 50   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 5           # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

model = word2vec.Word2Vec(sentences, workers=num_workers,
            size=num_features, min_count = min_word_count, window =
            context, sample = downsampling)

#model.most_similar('happy',  topn=15)
model_name = "model1"
model.save(model_name)


bigramer = gensim.models.Phrases(sentences)

bmodel = Word2Vec(bigramer[sentences], workers=num_workers,
            size=num_features, min_count = min_word_count, 
            window = context, sample = downsampling)

bmodel.save('bmodel3d')


##############################
#
# testing happy vs. sad
#
#####

happy = pd.read_csv('happy_counts.csv', index_col = 0)
sad = pd.read_csv('sad_counts.csv', index_col = 0)
np.sum(happy.mxm_id.isin(sad.mxm_id))
# Out[105]: 346
shareid = happy.mxm_id[happy.mxm_id.isin(sad.mxm_id)]
happy = happy[~happy.mxm_id.isin(shareid)]
sad = sad[~sad.mxm_id.isin(shareid)]

happy = happy.set_index('mxm_id')
sad = sad.set_index('mxm_id')
songw = {}
for i, song in happy.iterrows():
    songw[i] = pd.Series(list(filter(lambda x: x not in stop, song[song!=0].index[1:])))
for i, song in sad.iterrows():
    songw[i] = pd.Series(list(filter(lambda x: x not in stop, song[song!=0].index[1:])))


vocab = list(model.vocab.keys())
modeldf = pd.DataFrame(model.syn0)
modeldf.index = vocab

def findDistance(wlist, w, modeldf):
    wlistpos = modeldf.ix[wlist].sum()
    return np.linalg.norm(wlistpos - modeldf.ix[w])

output = [pd.Series([findDistance(words, 'happy', modeldf), findDistance(words, 'sad', modeldf)]) for k, words in songw.iteritems()]

output = pd.concat(output, axis = 1).T
output['mxm_id'] = songw.keys()
output['happy'] = (output[0] < output[1])
output['happy'] = (output[0] - output[1])/abs(output[0]) < -0.1
output['sad'] = (output[0] - output[1])/abs(output[1]) > 0.1

##############################
#
# word2vec + LSTM
#
#####

## load pre-trained word2vec dataset
from gensim.models.keyedvectors import KeyedVectors
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

#vocab = list(word2vec.vocab.keys())
#modeldf = pd.DataFrame(word2vec.syn0)
#modeldf.index = vocab
# stop = set(stopwords.words('english'))

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.engine import Input
from keras.preprocessing import sequence

weights = word2vec.syn0
layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
vocab = dict([(k, v.index) for k, v in word2vec.vocab.items()])
# input = Input(shape=(300,), dtype='float', name='input')

# model = Model(input=[input], output=[similarity])
# embeddings = layer(input)

## load matched song lyrics with tags
tagdf = pd.read_csv('Song_with_lyrics_new_for_new_tag.csv', index_col = 0)
tagdf2 = pd.read_csv('Song_with_lyrics_new_for_new_tag2.csv', index_col = 0)
tagdf3 = pd.read_csv('Song_with_lyrics_new_for_new_tag3.csv', index_col = 0)
tagdf = tagdf.append(tagdf2, ignore_index = True)
tagdf = tagdf.append(tagdf3, ignore_index = True)

tagdf = pd.read_csv('full_dataset.csv', index_col = 0)
newind = np.arange(tagdf.shape[0])
np.random.shuffle(newind)
## split out validation set
train_df = tagdf.ix[newind[:int(tagdf.shape[0]*0.9*0.7)]]
test_df = tagdf.ix[newind[int(tagdf.shape[0]*0.9*0.7):int(tagdf.shape[0]*0.9)]]
#traintest_df = tagdf.reset_index().ix[:int(tagdf.shape[0]*0.9)]
validate_df = tagdf.ix[newind[int(tagdf.shape[0]*0.9):]]

## split train/test set
# train_df = traintest_df.ix[:int(traintest_df.shape[0]*0.7)]
# test_df = traintest_df.ix[train_df.shape[0]:]
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

x_train = []
i = 0
for lyrics in train_df.Song_lyrics:
    wordlist = sentence_to_wordlist(lyrics)
    vecinput = [vocab[w] for w in wordlist if w in vocab.keys()]
    #vecinput = modeldf.ix[wordlist].dropna().values
    x_train.append(vecinput)
    i += 1
    if i%100 == 0:
        print (i)


x_test = []
i = 0
for lyrics in test_df.Song_lyrics:
    wordlist = sentence_to_wordlist(lyrics)
    vecinput = [vocab[w] for w in wordlist if w in vocab.keys()]
    # vecinput = modeldf.ix[wordlist].dropna().values
    # vecinput = vecinput.tolist()
    x_test.append(vecinput)
    i += 1
    if i%100 == 0:
        print (i)

x_train = np.array(x_train).reshape((train_df.shape[0],))
x_test = np.array(x_test).reshape((test_df.shape[0],))
x_train2 =sequence.pad_sequences(x_train, 150)
x_test2 =sequence.pad_sequences(x_test, 150)
print np.array([len(x) for x in x_train]).min()



from sklearn.preprocessing import LabelEncoder
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



alltags = tagdf.tag.unique()
tag = 'happy'
tagmap = {alltags[i]: i for i in range(len(alltags))}
train_df['tag_num'] = train_df.tag.map(lambda x: tagmap[x])
test_df['tag_num'] = test_df.tag.map(lambda x: tagmap[x])

model = Sequential()
#model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(layer)
model.add(LSTM(100))
#model.add(Dense(len(alltags), activation='sigmoid'))
model.add(Dense(len(alltags), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(x_train2, np.array(train_df.tag).reshape(train_df.shape[0],), validation_data=(x_test2, np.array(test_df.tag).reshape(test_df.shape[0],)), epochs = 3)
model.fit(x_train2, y_train, validation_data=(x_test2, y_test), epochs = 3)
#model = Model(input=[], output=[similarity])

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
