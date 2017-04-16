from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd
import numpy as np

from optparse import OptionParser
import sys
import csv

def feature_engineering(raw_data,output_file = "features.csv", feature_num = 250):
    print("Input song data from the lyrics_datafile...")
    with open(raw_data, 'rb') as f:
        reader = csv.reader(f)
        data_list = list(reader)
    
    data_list = np.array(data_list)
    lyrics = data_list[1:,7]
    tag = data_list[1:,5]

    print("Processing the input lyrics...")
    hv = HashingVectorizer(n_features=feature_num)
    trans = hv.transform(lyrics)
    # convert to dense matrix
    dense = trans.todense()
    dense = dense.tolist()
    for i in range(len(dense)):
        dense[i].append(tag[i])
    #print(len(dense[0]))
    print("Saving feature results...")
    with open(output_file, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(dense)
    print("-----Feature engineering DONE-----")



def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "Usage: %prog input_data.csv output.csv n_features \n input_data.csv is required (pass -h for more info)"
    parser = OptionParser(usage)

    parser.add_option("-f", "--outfile", dest="outfile",
                  help="output file to write to", metavar="FILE")
    parser.add_option("-n", "--n_features", dest="n_features", type = int,
                  help="num of features required")

    (options, args) = parser.parse_args(argv[1:])

    if len(args) < 1:
        parser.error("Must pass in a data file")
    if options.outfile is not None and options.n_features is not None:
        feature_engineering(args[0], output_file = options.outfile, feature_num = options.n_features)
    elif options.outfile is not None:
        feature_engineering(args[0], output_file = options.outfile)
    elif options.n_features is not None:
        feature_engineering(args[0], feature_num = options.n_features)
    else:
        feature_engineering(args[0])
    return 0

if __name__ == "__main__":
    sys.exit(main())