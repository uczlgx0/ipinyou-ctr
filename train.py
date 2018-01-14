from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import dump
from sklearn.metrics import accuracy_score
from time import time
from os.path import isdir, isfile, join
import pandas as pd
from csv import writer
from os import mkdir
from util import split_data

rf_models_path = 'rf_models.csv'
train_pct = 0.8

parser = ArgumentParser(description='Fits a Random forest model onto the training dataset and outputs performance',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t', '--trees', type=int, default=1000, help='The number of decision trees to use.')
parser.add_argument('-o', '--output', type=str, default='dump', help='Output directory to use.')
parser.add_argument('-i', '--input', type=int, default='data/ads_clean.csv', help='Path to the input dataset.')
args = parser.parse_args()
trees = args.trees
dump_dir = args.output

if not isdir(dump_dir):
    mkdir(dump_dir)

x = pd.read_csv(args.input)
x = x.as_matrix()
x = x.astype(int)

x_tr, x_te, y_tr, y_te = split_data(x, train_pct)

class_weights = {0: 1, 1: 1000}
rf = RandomForestClassifier(n_estimators=trees, n_jobs=-1, verbose=1, class_weight=class_weights)
rf.fit(x_tr, y_tr)

accuracy = accuracy_score(y_te, rf.predict(x_te))
print(f'Accuracy: {accuracy}')
id = str(int(time()))
if isfile(rf_models_path):
    with open(rf_models_path, 'a') as file:
        models_writer = writer(file)
        models_writer.writerow([id, accuracy, rf.n_estimators, rf.class_weight[0], rf.class_weight[1]])
else:
    with open(rf_models_path, 'w') as file:
        models_writer = writer(file)
        models_writer.writerow(
            ['id', 'accuracy', 'n_estimators', 'class_weight_0', 'class_weight_1'])
        models_writer.writerow([id, accuracy, rf.n_estimators, rf.class_weight[0], rf.class_weight[1]])

if not isdir(join(dump_dir, 'rf')):
    mkdir(join(dump_dir, 'rf'))
dump(rf, join(dump_dir, 'rf', id))
