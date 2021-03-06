import numpy as np
from time import time
from os import mkdir
from os.path import isfile, isdir, join
from csv import writer
from sklearn.externals.joblib import dump


def split_data(x, train_pct):
    """
    Splits observations in a training and test dataset.
    :param x: The observations as numpy array that contain class labels in the last row.
    :param train_pct: The percentage of observations to use for training. 1 - `training_pct` is used for the test set.
    :return: Four numpy arrays:
            1. Training observations without class labels
            2. Test observations without class labels
            3. Trainings class labels
            4. Test class labels
    """
    y_tr, y_te = None, None
    while y_te is None or 1 not in y_te:  # Ensure that test dataset does not contain solely negative classes
        training_size = int(len(x) * train_pct)
        indices = np.random.permutation(len(x))
        training_idx, test_idx = indices[:training_size], indices[training_size:]
        x_tr, x_te = x[training_idx, :], x[test_idx, :]
        y_tr, y_te = x_tr[:, 0], x_te[:, 0]
        x_tr, x_te = x_tr[:, 1:], x_te[:, 1:]  # First column is label column
    return x_tr, x_te, y_tr, y_te


def dump_rf(model, path, stat_path, kappa, precision, recall):
    id = str(int(time()))
    if isfile(stat_path):
        with open(stat_path, 'a') as file:
            models_writer = writer(file)
            models_writer.writerow(
                [id, kappa, recall, precision, model.n_estimators, model.class_weight[0], model.class_weight[1]])
    else:
        with open(stat_path, 'w') as file:
            models_writer = writer(file)
            models_writer.writerow(
                ['id', 'kappa', 'recall', 'precision', 'n_estimators', 'class_weight_0', 'class_weight_1'])
            models_writer.writerow(
                [id, kappa, recall, precision, model.n_estimators, model.class_weight[0], model.class_weight[1]])

    if not isdir(join(path, 'rf')):
        mkdir(join(path, 'rf'))

    dump(model, join(path, 'rf', id))