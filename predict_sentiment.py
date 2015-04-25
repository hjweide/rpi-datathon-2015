#!/usr/bin/env python

import cPickle as pickle
import numpy as np
import pandas as pd

from utils import load_imdb_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from time import time

retrain = False
if retrain:
    train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)

    clean_train_reviews = load_imdb_data('cleaned.pickle')

    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()

    kf = StratifiedKFold(train['sentiment'], round(1. / 0.2))

    train_indices, val_indices = next(iter(kf))

    train_data = train_data_features[train_indices]
    train_labels = train['sentiment'][train_indices]
    val_data = train_data_features[val_indices]
    val_labels = train['sentiment'][val_indices]

    forest = RandomForestClassifier(n_estimators=100)
    print('training random forest...')
    t0 = time()
    forest = forest.fit(train_data, train_labels)
    print('completed in %.2f seconds' % (time() - t0))
    with open('model.pickle', 'wb') as mfile, open('vectorizer.pickle', 'wb') as vfile:
        pickle.dump(forest, mfile)
        pickle.dump(vectorizer, vfile)
else:
    print('loading random forest...')
    with open('model.pickle', 'rb') as mfile, open('vectorizer.pickle', 'rb') as vfile:
        forest = pickle.load(mfile)
        vectorizer = pickle.load(vfile)

if False:
    print('predicting on training set...')
    train_pred = forest.predict(train_data)
    train_score = roc_auc_score(train_labels, train_pred)
    print('train score = %.6f' % (train_score))

    print('predicting on validation set...')
    val_pred = forest.predict(val_data)
    val_score = roc_auc_score(val_labels, val_pred)
    print('validation score = %.6f' % (val_score))

university_name = 'rpi'
with open('data/%s.pickle' % (university_name)) as ifile:
    all_list = pickle.load(ifile)

    print('predicting on %s set...' % university_name)
    for thread in all_list:
        submission, comments = thread[0], thread[1]
        data = np.vstack([submission] + comments).flatten()
        data_features = vectorizer.transform(data).toarray()
        univ_pred = forest.predict(data_features)
        for post, pred in zip(data, univ_pred):
            print('%s: %d' % (post, pred))
