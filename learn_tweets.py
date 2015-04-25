#!/usr/bin/env python

import cPickle as pickle
import numpy as np

from os import getcwd
from os.path import join
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from time import time
from utils import read_tweets


def learn_sentiment_from_tweets(clean_tweets, clean_tweets_sentiments, modelfile, retrain=False):
    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=1000)

    # compute the bag-of-words model on the data
    clean_tweets_features = vectorizer.fit_transform(clean_tweets).toarray()
    clean_tweets_sentiments = np.array(clean_tweets_sentiments)
    vocab = vectorizer.get_feature_names()  # NOQA

    # split into training and validation data
    kf = StratifiedKFold(clean_tweets_sentiment, round(1. / 0.2))
    train_indices, valid_indices = next(iter(kf))

    train_data = clean_tweets_features[train_indices]
    train_labels = clean_tweets_sentiments[train_indices]
    valid_data = clean_tweets_features[valid_indices]
    valid_labels = clean_tweets_sentiments[valid_indices]

    if retrain:
        print('training random forest...')
        forest = RandomForestClassifier(n_estimators=100)
        t0 = time()
        forest = forest.fit(train_data, train_labels)
        print('training completed in %.2f' % (time() - t0))
        print('saving model to %s' % (modelfile))
        with open(modelfile, 'wb') as ofile:
            pickle.dump(forest, ofile)
    else:
        print('loading random forest...')
        with open(modelfile, 'rb') as ifile:
            forest = pickle.load(ifile)

    print('predicting on training set...')
    train_pred = forest.predict(train_data)
    train_score = roc_auc_score(train_labels, train_pred)
    print('train score = %.6f' % (train_score))

    print('predicting on validation set...')
    valid_pred = forest.predict(valid_data)
    valid_score = roc_auc_score(valid_labels, valid_pred)
    print('validation score = %.6f' % (valid_score))


if __name__ == '__main__':
    root = getcwd()
    datafile = join(root, 'data', 'tweets_clean_short.csv')
    tweetsfile = join(root, 'data', 'tweets_clean_short.pickle')
    modelfile = join(root, 'data', 'model.pickle')
    clean_tweets, clean_tweets_sentiment = read_tweets(datafile, tweetsfile)
    learn_sentiment_from_tweets(clean_tweets, clean_tweets_sentiment, modelfile, retrain=False)
