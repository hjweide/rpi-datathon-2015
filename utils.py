#!/usr/bin/env python

import json

from os import getcwd
from os.path import join


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


# example usage
if __name__ == '__main__':
    root = getcwd()
    datafile = join(root, 'data', 'dummy.txt')

    business_list, review_list, user_list = load_data(datafile)

    print('businesses')
    for business in business_list:
        print('type: %s, id: %s' % (business['type'], business['business_id']))

    print('reviews')
    for review in review_list:
        print('type: %s, id: %s' % (review['type'], review['business_id']))
