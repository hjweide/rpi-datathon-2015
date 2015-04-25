#!/usr/bin/env python

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from os.path import join


months = {'01': 'Jan',
          '02': 'Feb',
          '03': 'Mar',
          '04': 'Apr',
          '05': 'May',
          '06': 'Jun',
          '07': 'Jul',
          '08': 'Aug',
          '09': 'Sep',
          '10': 'Oct',
          '11': 'Nov',
          '12': 'Dec'}


def predict_sentiment_posts(thread_list, model, vectorizer):
    sentiment_dict = defaultdict(list)
    for thread in thread_list:
        submission, comments, date, ups = thread
        month = date.split(',')[1]
        if len(comments) > 0:
            data = np.vstack([submission] + comments).flatten()
        else:
            data = np.array([submission])

        data_features = vectorizer.transform(data).toarray()
        pred = model.predict(data_features)

        for p in pred:
            sentiment_dict[month].append(p)

        #if len(comments) > 0:
        #    title_weight = 0.1
        #    title_sentiment = title_weight * pred[0]
        #    non_title_sentiment = (1 - title_weight) * np.mean(pred[1:])
        #    #title_sentiment = 0
        #    #non_title_sentiment = np.mean(pred)
        #else:
        #    title_sentiment = pred[0]
        #    non_title_sentiment = 0

        #thread_sent = title_sentiment + non_title_sentiment
        #sentiment_list.append(thread_sent)

    return sentiment_dict


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

    #plt.show()
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    with open(join('data', 'model.pickle'), 'rb') as ifile:
        model = pickle.load(ifile)
    with open(join('data', 'vectorizer.pickle'), 'rb') as ifile:
        vectorizer = pickle.load(ifile)

    university_names = ['mit', 'usc', 'uiuc', 'rpi', 'rit', 'uwaterloo']
    #university_names = ['mit']
    for university in university_names:
        key_list, sentiment_list = [], []
        print('processing %s' % university)
        with open(join('data', '%s.pickle' % university)) as ifile:
            post_list = pickle.load(ifile)
            sentiment_dict = predict_sentiment_posts(post_list, model, vectorizer)
            for key in sorted(sentiment_dict):
                total = len(sentiment_dict[key])
                positives = float(sentiment_dict[key].count(1)) / total
                negatives = float(sentiment_dict[key].count(0)) / total
                sentiment_list.append((positives, negatives))
                key_list.append(months[key])
                print('  %s: %.5f %.5f' % (months[key], positives, negatives))
            plot_bars(sentiment_list, key_list, university, join('plots', '%s.png' % university))
