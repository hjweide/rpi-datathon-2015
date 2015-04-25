#!/usr/bin/env python

import cPickle as pickle
import json
import pandas as pd
import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from os import getcwd
from os.path import exists, join


# loads a file containing one json object per line,
# similar to the Yelp academic dataset
def load_data(datafile):
    business_list, review_list, user_list = [], [], []
    with open(datafile, 'r') as ifile:
        for line in ifile:
            # if the json has single quotes
            # line = line.replace("'", '"')
            json_obj = json.loads(line)
            json_obj_type = json_obj['type']
            if json_obj_type == 'business':
                business_list.append(json_obj)
            elif json_obj_type == 'review':
                review_list.append(json_obj)
            elif json_obj_type == 'user':
                user_list.append(json_obj)
            else:
                print('json object of unknown type: %s' % (json_obj_type))

    return business_list, review_list, user_list


def load_imdb_data(filename):
    with open(filename, 'rb') as ifile:
        clean_data_reviews = pickle.load(ifile)

    return clean_data_reviews


# remove markup, stopwords, etc. from tweets
def tweet_to_words(tweet, min_length):
    tweet_text = BeautifulSoup(tweet).get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', tweet_text)

    # convert to lowercase and get rid of all extra whitespace
    words = letters_only.lower().split()

    # it is faster to search a set than a list
    stops = set(stopwords.words('english'))

    meaningful_words = [w for w in words if w not in stops]

    if len(meaningful_words) >= min_length:
        return ' '.join(meaningful_words)
    return None


# read tweets from the csv file
def read_tweets(datafile, filename, min_length=0):
    # check if we have already saved this file to disk to save computation
    if not exists(filename):
        # some lines are bad, just skip them
        data = pd.read_csv(datafile, header=0, delimiter=',', quotechar='"', error_bad_lines=False, encoding='utf-8-sig')

        num_tweets = data['ItemID'].size
        print('cleaned tweets will be saved to %s' % (filename))
        clean_tweets, clean_tweets_sentiment = [], []

        for i in xrange(num_tweets):
            if (i + 1) % 1000 == 0:
                print('  cleaning tweet %d of %d' % (i + 1, num_tweets))

            clean_tweet = tweet_to_words(data['SentimentText'][i], min_length)
            if clean_tweet is not None:
                clean_tweet_sentiment = data['Sentiment'][i]
                clean_tweets.append(clean_tweet)
                clean_tweets_sentiment.append(clean_tweet_sentiment)

        # save the cleaned tweets to disk for future use
        with open(filename, 'wb') as ofile:
            pickle.dump((clean_tweets, clean_tweets_sentiment), ofile)

    else:
        print('loading cleaned tweets from disk: %s' % (filename))
        with open(filename, 'rb') as ifile:
            clean_tweets, clean_tweets_sentiment = pickle.load(ifile)

    return clean_tweets, clean_tweets_sentiment


# example usage
if __name__ == '__main__':
    root = getcwd()
    datafile = join(root, 'data', 'dummy.txt')
    tweetsfile = join(root, 'data', 'tweets.csv')
    tweetsfile_clean = join(root, 'data', 'tweets_clean.pickle')

    business_list, review_list, user_list = load_data(datafile)

    clean_tweets, clean_tweets_sentiment = read_tweets(tweetsfile, tweetsfile_clean)
