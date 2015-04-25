#!/usr/bin/env python

import cPickle as pickle
import numpy as np

from os.path import join


def predict_sentiment_posts(thread_list, model, vectorizer):
    sentiment_list = []
    for thread in thread_list:
        submission, comments, date, ups = thread
        if len(comments) > 0:
            data = np.vstack([submission] + comments).flatten()
        else:
            data = np.array([submission])

        data_features = vectorizer.transform(data).toarray()
        pred = model.predict(data_features)

        if len(comments) > 0:
            #title_weight = 0.1
            #title_sentiment = title_weight * pred[0]
            #non_title_sentiment = (1 - title_weight) * np.mean(pred[1:])
            title_sentiment = 0
            non_title_sentiment = np.mean(pred)
        else:
            continue
            title_sentiment = pred[0]
            non_title_sentiment = 0

        thread_sent = title_sentiment + non_title_sentiment
        sentiment_list.append(thread_sent)

    return np.mean(sentiment_list)


if __name__ == '__main__':
    with open(join('data', 'model.pickle'), 'rb') as ifile:
        model = pickle.load(ifile)
    with open(join('data', 'vectorizer.pickle'), 'rb') as ifile:
        vectorizer = pickle.load(ifile)

    university_names = ['mit', 'usc', 'uiuc', 'rpi']
    for university in university_names:
        print('processing %s' % university)
        with open(join('data', '%s.pickle' % university)) as ifile:
            post_list = pickle.load(ifile)
            print predict_sentiment_posts(post_list, model, vectorizer)
