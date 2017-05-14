import csv
import numpy as np
import pandas as pd


def mostSimilarFromList(inputw, wlist, word2vec):
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


def tag_prediction(model, tag, raw_data="playlist.csv"):
    print("Input song data from the lyrics_datafile...")
    with open(raw_data, 'rb') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    data_list = np.array(data_list)

    df = pd.read_csv(raw_data, index_col=0)
    df_no_dup = df.drop_duplicates('Song_txt_loc')
    if tag in df_no_dup.columns:
        sorted_df = df_no_dup.sort_values(tag, ascending=False)
        name = sorted_df[:10].Name.values
        artist = sorted_df[:10].Aritist.values
        link = sorted_df[:10].Song_link.values
        tag_p = sorted_df[:10][tag].values
        d = {'name': list(name), 'artist': list(artist), 'tag_p': list(tag_p), 'link': list(link)}
        return d
    else:
        print("---- Tag Is Not in the Database, Generating Most Similar Tag ----")
        sim_tag = mostSimilarFromList(tag, list(df_no_dup.columns[6:-4]), model)
        print("---- Done Generating the Most Similar Tag ----")
        if sim_tag is None:
            print("Sorry! Do not have songs for this tag.")
            return None
        else:
            sorted_df = df_no_dup.sort_values(sim_tag, ascending=False)
            name = sorted_df[:10].Name.values
            artist = sorted_df[:10].Aritist.values
            link = sorted_df[:10].Song_link.values
            tag_p = sorted_df[:10][sim_tag].values
            d = {'name': list(name), 'artist': list(artist), 'tag_p': list(tag_p), 'link': list(link)}
            return d
