#!/usr/bin/env python

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from os.path import join

months = {'01': 'Jan 15',
          '02': 'Feb 15',
          '03': 'Mar 15',
          '04': 'Apr 15',
          '05': 'May 14',
          '06': 'Jun 14',
          '07': 'Jul 14',
          '08': 'Aug 14',
          '09': 'Sep 14',
          '10': 'Oct 14',
          '11': 'Nov 14',
          '12': 'Dec 14'}


# given a trained model and words-to-feature encoder, returns the predicted sentiment
# of each submission/comment in that subreddit for each month
def predict_sentiment_posts(thread_list, model, vectorizer):
    sentiment_dict = defaultdict(list)
    for thread in thread_list:
        submission, comments, date, ups = thread
        month = date.split(',')[1]
        # some submissions don't have any comments
        if len(comments) > 0:
            data = np.vstack([submission] + comments).flatten()
        else:
            data = np.array([submission])

        data_features = vectorizer.transform(data).toarray()
        pred = model.predict(data_features)

        for p in pred:
            sentiment_dict[month].append(p)

    return sentiment_dict


# plot the bar graph of positive and negative sentiment for each month
def plot_bars(values_list, key_list, title, filename):
    N = len(values_list)
    ind = np.arange(N)

    width = 0.35
    plots = []
    fig, ax = plt.subplots()

    color_list = ['green', 'red']
    values_list_transpose = [tup for tup in zip(*values_list)]
    for i, (values, color) in enumerate(zip(values_list_transpose, color_list)):
        plots.append(ax.bar(ind + i * width, values, width, color=color))

    ax.set_title(title)
    ax.set_ylim((0.0, 1.0))
    ax.set_xticks(ind + width)
    ax.set_xticklabels(key_list)
    ax.legend(plots, ['positive', 'negative'])

    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    # load the pre-trained model
    with open(join('data', 'model.pickle'), 'rb') as ifile:
        model = pickle.load(ifile)
    # load the pre-trained words-to-feature encoder
    with open(join('data', 'vectorizer.pickle'), 'rb') as ifile:
        vectorizer = pickle.load(ifile)

    # the names of each university's subreddit
    university_names = ['mit', 'usc', 'uiuc', 'rpi', 'rit', 'uwaterloo', 'cornell', 'UTAustin', 'ucla']
    for university in university_names:
        key_list, sentiment_list = [], []
        print('processing %s' % university)
        with open(join('data', '%s.pickle' % university)) as ifile:
            post_list = pickle.load(ifile)
            sentiment_dict = predict_sentiment_posts(post_list, model, vectorizer)
            for key in sorted(sentiment_dict):
                total = len(sentiment_dict[key])
                positives = sentiment_dict[key].count(1)
                negatives = sentiment_dict[key].count(0)
                sentiment_list.append((positives, negatives))
                key_list.append(months[key])
                print('  %s: %d %d' % (months[key], positives, negatives))
            plot_bars(sentiment_list, key_list, university, join('plots', '%s.png' % university))
