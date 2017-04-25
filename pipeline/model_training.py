from sklearn.ensemble import RandomForestClassifier
import cPickle
import sys
from optparse import OptionParser
import csv
import numpy as np

def model_train(feature_data, n_est = 15, max_dep =5):
    print("Input features from the feature database...")
    with open(feature_data, 'rb') as f:
        reader = csv.reader(f)
        data_list = list(reader)
    data_list = np.array(data_list)

    train_x = data_list[:,:-1]
    train_y = data_list[:,-1]
    print("Running Model...")
    rf = RandomForestClassifier(n_estimators = n_est, max_depth =max_dep,class_weight = "balanced")
    rf.fit(train_x, train_y)

    with open('model.pickle', 'wb') as f:
        cPickle.dump(rf, f)
    print("-----Modeling Done-----")
# RF_short = RandomForestClassifier(n_estimators = 15, max_depth =5)

def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "Usage: %prog input_feature.csv (pass -h for more info)"
    parser = OptionParser(usage)

    parser.add_option("-e", "--n_estimator", dest="n_estimator",
                  help="number of estimators required for Random Forest Model", type = int)
    parser.add_option("-d", "--max_depth", dest="max_depth", type = int,
                  help="max depth of the Random Forest model")

    (options, args) = parser.parse_args(argv[1:])

    if len(args) < 1:
        parser.error("Must pass in a feature file")
    
    if options.max_depth is not None and options.n_estimator is not None:
        model_train(args[0], n_est = options.n_estimator, max_dep = options.max_depth)
    elif options.n_estimator is not None:
        model_train(args[0], n_est = options.n_estimator)
    elif options.max_depth is not None:
        model_train(args[0], max_dep = options.max_depth)
    else:
        model_train(args[0])
    return 0

if __name__ == "__main__":
    sys.exit(main())