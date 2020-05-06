import Models

import pdb
import matplotlib.pyplot as plt
import numpy as np
import os
from process_skeleton import classes,normalize_skeleton, vids_with_missing_skeletons, draw_skeleton
from keras.regularizers import l2,l1
from visualize_filters import extract_feature_from_file_index as ef

#Parameters
n_classes = 60
feat_dim = 150
max_len = 300

weights = '/Users/aviad/PycharmProjects/TCNActionRecognition/weights/cross_view.hdf5'

def single_classification(ind):

    model = Models.TCN_simple_resnet(n_classes,
                              feat_dim,
                              max_len,
                              gap=1,
                              dropout=0.0,
                              kernel_regularizer=l1(1.e-4),
                              activation="relu")

    model.load_weights(weights)

    x, y, y_name, file_path = ef(ind)
    X = np.zeros((max_len, feat_dim))
    X[:x.shape[0]] = x
    # pdb.set_trace()
    X = X.reshape((1, X.shape[0], X.shape[1]))
    prediction = np.argmax(model.predict(X))

    print("Ground Truth: ", y, "(%s)" % y_name)
    print("Prediction: ", prediction, "(%s)" % classes[prediction])

    if y == prediction:
        return 1
    else:
        return 0

def classification_score():
    bad_files = vids_with_missing_skeletons()
    skeleton_dir_root = "/Users/aviad/PycharmProjects/TCNActionRecognition/Data/nturgb+d_skeletons"
    skeleton_files = os.listdir(skeleton_dir_root)
    count = 0
    count_all = 0
    for indx in range(0,len(skeleton_files)):

        file_name = skeleton_files[indx]
        if file_name in bad_files:
            continue
        else:
            count += single_classification(indx)
            count_all +=count_all
    return count/count_all

if __name__ == "__main__":
    # single_classification(200)
    classification_score()