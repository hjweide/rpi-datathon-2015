#!/usr/bin/env python

import cPickle as pickle
import datetime
import praw
import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords


# convert unix time to Eastern Daylight Time
def unix_to_edt(unix_time):
    return datetime.datetime.fromtimestamp(int(unix_time)).strftime('%Y,%m,%d')


# convert a post into words that can be used by the learning model
def clean_post(raw_text):
    post_text = BeautifulSoup(raw_text).get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', post_text)

    # convert to lowercase and get rid of all extra whitespace
    words = letters_only.lower().split()

    # it is faster to search a set than a list in Python
    stops = set(stopwords.words('english'))

    meaningful_words = [w for w in words if w not in stops]

    return ' '.join(meaningful_words)


if __name__ == '__main__':
    user_agent = ''         # user-agent for connecting to the reddit api goes here
    university_name = ''    # name of the subreddit that should be scraped goes here

    target_subreddit = university_name
    r = praw.Reddit(user_agent=user_agent)
    sub = r.get_subreddit(target_subreddit)

    # praw respects the reddit api by limiting calls to 1 every 2 seconds
    submissions = sub.get_top_from_year(limit=3000)

    all_list = []
    for submission in submissions:
        # current submission's creation time
        utc_time = submission.created_utc

        date_created = unix_to_edt(utc_time)
        cleaned_submission = clean_post(submission.title)

        # the number of upvotes a thread received
        ups = submission.ups

        print('%s %s %d' % (cleaned_submission, date_created, ups))
        comments = submission.comments

        # parse the tree of comments and their replies, ignore ordering
        flat_comments = praw.helpers.flatten_tree(comments)
        cleaned_comments = [clean_post(str(comment)) for comment in flat_comments]

        all_list.append((cleaned_submission, cleaned_comments, date_created, ups))

    # dump this list-of-tuples to a pickle object for reading by the learning model
    with open('%s.pickle' % (university_name), 'wb') as ofile:
        pickle.dump(all_list, ofile)
