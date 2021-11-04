import os
import argparse
import pickle

import cv2
import numpy as np
import torch
import pandas as pd
from skimage.color import rgb2hed
from sklearn.linear_model import LogisticRegression

def read_and_get_staining_ch(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hed = rgb2hed(img)
    d = img_hed[:,:,2]
    return d

def extract_max_avg_staining_intensity(img_path, patch_size = 224):
    d = read_and_get_staining_ch(img_path)
    tensor = torch.tensor(d)
    dims = tensor.dim()
    tensor_unfold = tensor.unfold(dims-2, size = patch_size, step = patch_size).unfold(dims-1, size = patch_size, step = patch_size)
    avg_unfold = torch.mean(tensor_unfold, dim = (2,3))
    max_avg_intensity = np.max(avg_unfold.numpy())
    return max_avg_intensity

def get_args():
    parser = argparse.ArgumentParser(description='Train and test staining intensity classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', dest='task', type=str, default='her2-status', help='ihc-score or her2-status')
    parser.add_argument('--train_csv', dest='train_csv', type=str, default='train.csv', help='.csv file containing the training examples')
    parser.add_argument('--test_csv', dest='test_csv', type=str, default='test.csv', help='.csv file containing the test examples')
    parser.add_argument('--pickle_dir', dest='pickle_dir', type=str, default='./', help='Folder to store the pickled classifier')
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='./out', help='Path to save the inference results')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.pickle_dir, exist_ok = True)
    os.makedirs(args.out_dir, exist_ok = True)

    train_df = pd.read_csv(args.train_csv)
    targets = train_df[args.task].values

    x_train = train_df.apply(lambda x: extract_max_avg_staining_intensity(x['filename']), axis = 1).values.reshape(-1,1)
    clf = LogisticRegression(max_iter = 200, class_weight = 'balanced').fit(x_train, targets)
    pickle.dump(clf, open(os.path.join(args.pickle_dir, '{}_staining_intensity_classifier.pkl'.format(args.task)), 'wb'))

    test_df = pd.read_csv(args.test_csv)
    x_test = test_df.apply(lambda x: extract_max_avg_staining_intensity(x['filename']), axis = 1).values.reshape(-1,1)

    predictions = clf.predict(x_test)
    results_df = pd.concat([
        test_df['filename'], 
        test_df[args.task].rename('true_class'), 
        pd.Series(data = predictions, name = 'predicted_class')
        ], axis = 1)

    results_df.to_csv(os.path.join(args.out_dir,'{}_staining_intensity_inference_out.csv'.format(args.task)), index = False)