from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import cPickle
import sys
from optparse import OptionParser
import csv
import numpy as np
import pandas as pd

import nltk
from nltk import word_tokenize

def tag_prediction(raw_data = "multiclassLSTMOutput.csv", tag = "all"):
    print("Input song data from the lyrics_datafile...")
    with open(raw_data, 'rb') as f:
        reader = csv.reader(f)
        data_list = list(reader)
    
    data_list = np.array(data_list)

    if tag == "all":
        index = np.random.randint(0,len(data_list)-1, 10)
        l = [ data_list[i] for i in index ]
        print("-----Prediction Done-----")
        print("")
        print("The prediction results are:")
        name = []
        artist = []
        for i in range(10):
            id,n,art = l[i][1],l[i][2],l[i][3]
            name.append(n)
            artist.append(art)
        return zip(name,artist)
    else:
        df = pd.read_csv("./playlist.csv", index_col=0)
        df_no_dup = df.drop_duplicates('Song_txt_loc')
        if tag in df_no_dup.columns:
            sorted_df = df_no_dup.sort_values(tag,ascending=False)
            name = sorted_df[:10].Name.values
            artist = sorted_df[:10].Aritist.values
            link = sorted_df[:10].Song_link.values
            tag_p = sorted_df[:10][tag].values
            return zip(name, artist,tag_p,link)
        else:
            return "Sorry! Do not have songs for this tag."


def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "Usage: %prog -t tag (pass -h for more info)"
    parser = OptionParser(usage)

    parser.add_option("-t", "--tag", dest="tag",
                  help="Tag of the songs we want")


    (options, args) = parser.parse_args(argv[1:])

    if options.tag is not None:
        tag = tag_prediction(tag= options.tag)
    else:
        tag = tag_prediction()
    if type(tag) == list:
        print("-----Prediction Done-----")
        print("")
        print("The prediction results are:")
        print("Name:\t\t\tArtist:")
        for i in range(len(tag)):
            n,a,t,l = tag[i]
            print(n+"\t"+a+"\t"+str(t)+"\t"+l)
    else:
        print tag
    return 0

if __name__ == "__main__":
    sys.exit(main())