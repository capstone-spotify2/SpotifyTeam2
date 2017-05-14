import sys
from optparse import OptionParser
import numpy as np
import pandas as pd

from gensim.models.keyedvectors import KeyedVectors as KV


def mostSimilarFromList(inputw, wlist):
    word2vec = KV.load_word2vec_format('../1-LSTM_model_for_tag_prediction/GoogleNews-vectors-negative300.bin',
                                       binary=True)
    inputw = inputw.split()
    inputw_ = "_".join(inputw)
    vocabs = word2vec.vocab.keys()
    if inputw_ in vocabs:
        sims = [word2vec.similarity(inputw_, w) for w in wlist]
        return wlist[np.argmax(sims)]
    elif (np.sum([iw in vocabs for iw in inputw]) > 0) & (len(inputw) > 1):
        sims = []
        for w in wlist:
            sims.append(np.mean([word2vec.similarity(iw, w)
                                 for iw in inputw if iw in word2vec.vocab.keys()]))
        return wlist[np.argmax(sims)]
    else:
        print ("input not in vocabulary")
        return None


def tag_prediction(tag, raw_data="playlist.csv"):
    print("Input song data from the lyrics_datafile...")

    df = pd.read_csv(raw_data, index_col=0)
    df_no_dup = df.drop_duplicates('Song_txt_loc')
    if tag in df_no_dup.columns:
        sorted_df = df_no_dup.sort_values(tag, ascending=False)
        name = sorted_df[:10].Name.values
        artist = sorted_df[:10].Aritist.values
        link = sorted_df[:10].Song_link.values
        return zip(name, artist, link)
    else:
        print("---- Tag Is Not in the Database, Generating Most Similar Tag ----")
        sim_tag = mostSimilarFromList(tag, list(df_no_dup.columns[6:-4]))
        print("---- Done Generating the Most Similar Tag ----")
        if sim_tag is None:
            return "Sorry! Do not have songs for this tag."
        else:
            sorted_df = df_no_dup.sort_values(sim_tag, ascending=False)
            name = sorted_df[:10].Name.values
            artist = sorted_df[:10].Aritist.values
            link = sorted_df[:10].Song_link.values
            return zip(name, artist, link)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "Usage: %prog tag (pass -h for more info)"
    parser = OptionParser(usage)

    (options, args) = parser.parse_args(argv[1:])

    if len(args) < 1:
        parser.error("Must pass in a tag name")

    tag = tag_prediction(args[0])

    if type(tag) == list:
        print("-----Prediction Done-----")
        print("")
        print("The prediction results are:")
        print("Name:\t\t\tArtist:")
        for i in range(len(tag)):
            n, a, t, l = tag[i]
            print(n + "\t" + a + "\t" + str(t) + "\t" + l)
    else:
        print tag
    return 0


if __name__ == "__main__":
    sys.exit(main())
