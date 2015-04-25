#!/usr/bin/env python

import cPickle as pickle
import datetime
import praw
import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

user_agent = 'scraping-reddit-k1'
university_name = 'UTAustin'


def unix_to_edt(unix_time):
    return datetime.datetime.fromtimestamp(int(unix_time)).strftime('%Y,%m,%d')


def clean_post(raw_text):
    post_text = BeautifulSoup(raw_text).get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', post_text)

    # convert to lowercase and get rid of all extra whitespace
    words = letters_only.lower().split()

    # it is faster to search a set than a list
    stops = set(stopwords.words('english'))

    meaningful_words = [w for w in words if w not in stops]

    return ' '.join(meaningful_words)


target_subreddit = university_name
r = praw.Reddit(user_agent=user_agent)
sub = r.get_subreddit(target_subreddit)
submissions = sub.get_top_from_year(limit=3000)

all_list = []
for submission in submissions:
    # current submission's creation time
    utc_time = submission.created_utc

    date_created = unix_to_edt(utc_time)
    cleaned_submission = clean_post(submission.title)
    ups = submission.ups

    print('%s %s %d' % (cleaned_submission, date_created, ups))
    comments = submission.comments
    flat_comments = praw.helpers.flatten_tree(comments)
    cleaned_comments = [clean_post(str(comment)) for comment in flat_comments]

    all_list.append((cleaned_submission, cleaned_comments, date_created, ups))

with open('%s.pickle' % (university_name), 'wb') as ofile:
    pickle.dump(all_list, ofile)
