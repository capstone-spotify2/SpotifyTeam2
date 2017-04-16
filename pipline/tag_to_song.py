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

def tag_prediction(raw_data, tag = "all"):
    print("Input song data from the lyrics_datafile...")
    with open(raw_data, 'rb') as f:
        reader = csv.reader(f)
        data_list = list(reader)
    
    data_list = np.array(data_list)

    if tag == "all":
        index = np.random.randint(0,len(data_list)-1, 5)
        l = [ data_list[i] for i in index ]
        print("-----Prediction Done-----")
        print("")
        print("The prediction results are:")
        for i in range(5):
            id,name,art = l[i][2],l[i][3],l[i][4]
            print(id+ " "+name+ ", "+art)
    else:
        df = pd.read_csv(raw_data)
        tag_df = df[df['tag'].isin([tag])].values
        index = np.random.randint(0,len(tag_df)-1, 5)
        l = [ tag_df[i] for i in index ]
        print("-----Prediction Done-----")
        print("")
        print("The prediction results are:")
        for i in range(5):
            id,name,art = l[i][2],l[i][3],l[i][4]
            print(id+ " "+name+ ", "+art)

    
    
# RF_short = RandomForestClassifier(n_estimators = 15, max_depth =5)

def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "Usage: %prog input_data.csv -t tag \n input_data.csv is required (pass -h for more info)"
    parser = OptionParser(usage)

    parser.add_option("-t", "--tag", dest="tag",
                  help="Tag of the songs we want")


    (options, args) = parser.parse_args(argv[1:])

    if len(args) < 1:
        parser.error("Must pass in a data file")
    if options.tag is not None:
        tag_prediction(args[0], tag= options.tag)
    else:
        tag_prediction(args[0])
    return 0

if __name__ == "__main__":
    sys.exit(main())