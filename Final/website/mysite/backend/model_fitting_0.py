import sys
from optparse import OptionParser
import csv
import numpy as np
import json

import nltk
from nltk import word_tokenize

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.engine import Input
from keras.preprocessing import sequence

from keras.utils import np_utils
import re
import pandas as pd
import nltk.data
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
import gensim
#from gensim.models import Word2Vec
#from gensim.models import word2vec
from gensim.models import Phrases
import logging
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model


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

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)
def LyricsToSentences(lyrics):
    lyrics = strip_non_ascii(lyrics)
    if '\r' in lyrics:
        lines = lyrics.split('\r')
        for i, l in enumerate(lines):
            if "<i>" in l:
                lines[i] = " "
        lines = ' '.join(lines).replace("\n \n", "\n").split(' \n')
        l = ' '
        for i in lines:
            l+= i
        return l
    elif "\n\n" in lyrics:
        lines = lyrics.split('\n')
        for i, l in enumerate(lines):
            if "<i>" in l:
                lines[i] = " "
        lines = ' '.join(lines).split('\r')
        l = ' '
        for i in lines:
            l+= i
        return l

def model_fit(input_song, model):
    print("Process the Input Song...")
    print input_song
    # with open(input_song, 'rb') as f:
    #     fa = LyricsToSentences(f.read())
    delete_table = "\xc3\xa2\xc2\x80\xc2\x99"
    lyrics = "".join(c for c in input_song if c not in delete_table)
    with open('vocab.json') as json_data:
        vocab = json.load(json_data)
        json_data.close()


    print("---- Convert Raw Lyrics to Word Vectors ----")
    wordlist = sentence_to_wordlist(lyrics)
    vecinput = [vocab[w] for w in wordlist if w in vocab]

    print("---- Running LSTM Model ----")
    #model = load_model(model)
    x_train2 =sequence.pad_sequences([vecinput], 150)
    pred = model.predict(x_train2)
    tag = np.array(['christmas', 'dark', 'emotional', 'energetic', 'freedom', 'funny',
           'grunge', 'halloween', 'happy', 'love', 'memory', 'party',
           'political', 'rain', 'relax', 'religious', 'sad'])
    preddf = pd.DataFrame(pred, columns = tag)
    sorted_preddf = preddf.T.sort_values(by=0,ascending = False).T
    tag = list(sorted_preddf.columns[:3])
    print (tag)
    prob = sorted_preddf.values[0][:3]
    print (prob)
    # print tag, prob
    return tag, prob


def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "Usage: %prog song_input.txt (pass -h for more info)"
    parser = OptionParser(usage)

    (options, args) = parser.parse_args(argv[1:])

    if len(args) < 1:
        parser.error("Must pass in a feature file")
    model = load_model('multiclass_model.h5')
    with open(argv[1], 'rb') as f:
        fa = LyricsToSentences(f.read())
    prediction = model_fit(fa,model)
    print "The three top tags are:"
    for tag, prob in prediction:
        print "tag: "+ tag +"\t prob: "+ str(prob)

    return 0

if __name__ == "__main__":
    sys.exit(main())
