
import sys
from optparse import OptionParser
import re
import pandas as pd
import json


from nltk.corpus import stopwords
import numpy as np


def sentence_to_wordlist(sentence, remove_stopwords=False):
    # 1. Remove non-letters
    sentence_text = re.sub(r'[^\w\s]', '', sentence)
    # 2. Convert words to lower case and split them
    words = sentence_text.lower().split()
    if remove_stopwords:
        stop = set(stopwords.words('english'))
        filter(lambda x: x not in stop, stop)
    # 3. Return a list of words
    return(words)


###############
#
# word2vec + LSTM
#
###


def build_model(output_model_name):
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence
    from gensim.models.keyedvectors import KeyedVectors

    print("---- Load in Trained Word2vec Model ----")

    # load pre-trained word2vec dataset
    word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    weights = word2vec.syn0
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
    vocab = dict([(k, v.index) for k, v in word2vec.vocab.items()])
    json_file = json.dumps(vocab)
    f = open("vocab.json", "w")
    f.write(json_file)
    f.close()

    print("---- Load in Lyrics Database ----")
    # load matched song lyrics with tags
    tagdf = pd.read_csv('../0-baseline_model_for_tag_prediction/full_dataset.csv', index_col=0)
    newind = np.arange(tagdf.shape[0])
    # randomly shuffle
    np.random.shuffle(newind)

    print("---- Train/Test Set Split ----")
    # split out training, testing, and validation set
    train_df = tagdf.ix[newind[:int(tagdf.shape[0]*0.9*0.7)]]
    test_df = tagdf.ix[newind[int(tagdf.shape[0]*0.9*0.7):int(tagdf.shape[0]*0.9)]]

    print("---- Convert Raw Lyrics to Word Vectors ----")
    x_train = []
    i = 0
    for lyrics in train_df.Song_lyrics:
        wordlist = sentence_to_wordlist(lyrics)
        vecinput = [vocab[w] for w in wordlist if w in vocab]
        x_train.append(vecinput)
        i += 1
        # if i%100 == 0:
        #     print i,

    x_test = []
    i = 0
    for lyrics in test_df.Song_lyrics:
        wordlist = sentence_to_wordlist(lyrics)
        vecinput = [vocab[w] for w in wordlist if w in vocab]
        x_test.append(vecinput)
        i += 1
        # if i%100 == 0:
        #     print i,

    x_all = []
    i = 0
    for lyrics in tagdf.Song_lyrics:
        wordlist = sentence_to_wordlist(lyrics)
        vecinput = [vocab[w] for w in wordlist if w in vocab]
        x_all.append(vecinput)
        i += 1
        # if i%100 == 0:
        #     print i,

    x_train = np.array(x_train).reshape((train_df.shape[0],))
    x_test = np.array(x_test).reshape((test_df.shape[0],))
    x_train2 = sequence.pad_sequences(x_train, 150)
    x_test2 = sequence.pad_sequences(x_test, 150)

    x_all = np.array(x_all).reshape((tagdf.shape[0],))

    alltags = tagdf.tag.unique()

    print("---- Convert y Values ----")
    # for multiclass model
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

    print("---- Start Training the LSTM Model ----")
    model = Sequential()
    model.add(layer)
    model.add(LSTM(100))
    model.add(Dense(len(alltags), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train2, y_train, validation_data=(x_test2, y_test), epochs=3)
    model.save(output_model_name+'.h5')

    print("---- Done Model Training ----")


def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "Usage: %prog output_model_name (pass -h for more info)"
    parser = OptionParser(usage)

    (options, args) = parser.parse_args(argv[1:])

    if len(args) < 1:
        parser.error("Must pass in an output model name")

    else:
        build_model(args[0])


if __name__ == "__main__":
    sys.exit(main())
