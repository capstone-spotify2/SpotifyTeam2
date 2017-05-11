
from feature_engineering import *
from model_fitting import *
from model_training import *
from tag_to_song import *
def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "Usage: %prog -f database.csv -s song.txt(pass -h for more info)"
    parser = OptionParser(usage)

    parser.add_option("-f", "--database.csv", dest="database",
                  help="The database we need for modeling")
    parser.add_option("-s", "--song.txt", dest="song_lyrics",
                  help="The lyrcis to classify")


    (options, args) = parser.parse_args(argv[1:])

    if options.database is not None and options.song_lyrics is not None:
    	feature_engineering(options.database,feature_num = 150)
        model_train('features.csv',n_est = 20, max_dep =10)
        with open('model.pickle', 'rb') as f:
            rf = cPickle.load(f)
        model_fit(options.song_lyrics, rf, feature_num = 150)
    elif options.database is not None:
        feature_engineering(options.database,feature_num = 150)
        model_train('features.csv',n_est = 20, max_dep =10)
    elif options.song_lyrics is not None:
    	#feature_engineering('full_dataset.csv',feature_num = 150)
        #model_train('features.csv',n_est = 20, max_dep =10)
        with open('model.pickle', 'rb') as f:
            rf = cPickle.load(f)
        model_fit(options.song_lyrics, rf, feature_num = 150)
    else:
        feature_engineering('full_dataset.csv',feature_num = 150)
        model_train('features.csv',n_est = 20, max_dep =10)



if __name__ == "__main__":
    sys.exit(main())