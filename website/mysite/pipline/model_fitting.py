from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import cPickle
import sys
from optparse import OptionParser
import csv
import numpy as np

import nltk
from nltk import word_tokenize

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

def model_fit(input_song, model, feature_num = 250):
    print("Process the input song...")
    with open(input_song, 'rb') as f:
        fa = LyricsToSentences(f.read())
    delete_table = "\xc3\xa2\xc2\x80\xc2\x99"
    lyrics = "".join(c for c in fa if c not in delete_table)
    lyrics = [lyrics]

    hv = HashingVectorizer(n_features=feature_num)
    trans = hv.transform(lyrics)
    # convert to dense matrix
    dense = trans.todense()
    dense = dense.tolist()
    res = []
    print("Running Model...")
    rank = zip(model.classes_, model.predict_proba(dense)[0])
    print("-----Prediction Done-----")
    print("")
    print("The prediction results are:")
    for i in range(3):
        tag,prob = sorted(rank, key=lambda x: -x[1])[i]
        print tag
        res.append(tag)
    return res

    
# RF_short = RandomForestClassifier(n_estimators = 15, max_depth =5)

def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "Usage: %prog song_input.txt (pass -h for more info)"
    parser = OptionParser(usage)

    parser.add_option("-n", "--n_features", dest="n_features", type = int,
                  help="num of features required")

    (options, args) = parser.parse_args(argv[1:])

    if len(args) < 1:
        parser.error("Must pass in a feature file")
    if options.n_features is not None:
        with open('model.pickle', 'rb') as f:
            rf = cPickle.load(f)
        model_fit(args[0], rf, feature_num = options.n_features)
    else:
        with open('model.pickle', 'rb') as f:
            rf = cPickle.load(f)
        model_fit(args[0], rf)

    
    return 0

if __name__ == "__main__":
    sys.exit(main())
