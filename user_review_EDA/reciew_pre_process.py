#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:54:57 2017

@author: vishwas
"""

import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import nltk

# Function to get the reviews only from the dataset in the form of str. 
def get_review_full():
    df1 = pd.read_csv("../raw_data/Beer_review_data/beerreview1_13.csv", usecols=["review"])
    df2 = pd.read_csv("../raw_data/Beer_review_data/beerreview14_26.csv", usecols=["review"])
    df3 = pd.read_csv("../raw_data/Beer_review_data/beerreview27_39.csv", usecols=["review"])
    df4 = pd.read_csv("../raw_data/Beer_review_data/beerreview40_51.csv", usecols=["review"])
    df = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)
    return ['' if type(x) is not str else x for x in list(df['review'])]

# Generating the random number of reviews from the reviews. 
def get_random_review(n = 20, seed = 0):
    '''input: n = number of review you want to generate, seed = random seed
       output: list, with each element being a review str'''
    assert n > 0, 'number of reviews must be greater than 0'
    assert n <= 100, 'don\'t do more than 100...'
    assert type(n) is int, 'number of reviews must be an integer'
    lst0 = range(4)
    lst1 = ['1_13.csv', '14_26.csv', '27_39.csv', '40_51.csv']
    lst2 = [111426, 68132, 61043, 36093]
    np.random.seed(seed)
    r0 = int(np.random.choice(lst0, size = 1)[0])
    r1 = lst1[r0]
    r2 = list(np.random.choice(range(lst2[r0]), size = n, replace = False))
    rev = list(pd.read_csv('../raw_data/Beer_review_data/beerreview' + r1, usecols = ['review'])['review'])
    rev = ['' if type(x) is not str else x for x in rev]
    return [rev[x] for x in r2]

# Function for generating the beer_name in a list. 
def get_beername_full():
    '''output: list of beer names. Each element is a beer name'''
    df1 = pd.read_csv("../raw_data/Beer_review_data/beerreview1_13.csv", usecols=['beer_name'])
    df2 = pd.read_csv("../raw_data/Beer_review_data/beerreview14_26.csv", usecols=['beer_name'])
    df3 = pd.read_csv("../raw_data/Beer_review_data/beerreview27_39.csv", usecols=['beer_name'])
    df4 = pd.read_csv("../raw_data/Beer_review_data/beerreview40_51.csv", usecols=['beer_name'])
    df = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)
    print 'beer name loaded'
    print '='*100
    return list(df['beer_name'])

def clean_str(s):
    s = s.lower().replace('\xe2\x80\x99', '\'').replace('|', ' ').replace('\r', ' ').replace('\n', ' ').replace('/', ' ').replace('@', ' ')
    s = re.sub('[.:\', \-!;"()?]', ' ', s)
    lst = re.sub('\s+', ' ', s).strip().split(' ')
    stop_words = set(stopwords.words('english')+ stopwords.words('german') +['&', ''])
    lst = [word for word in lst if word not in stop_words]
    lmtzr = WordNetLemmatizer()
    lst = map(lambda x: lmtzr.lemmatize(x.decode ('utf-8', 'ignore')).encode('utf-8', 'ignore'), lst)
    lst = [word for word in lst if word != '']
    return lst

def review_clean_full():
    l = get_review_full()
    result= []
    i = 0
    for review in l:
        result.append(clean_str(review))
        
        i += 1
    return result

# Function for creating a nested list of beer names
def names_tuples(name, z):
    ls =[]
    for i in z:
        if i[0] == name:
            ls.append(i[1])
    return [item for sublist in ls for item in sublist]

def word_group(name_lst, rev_lst):
    assert len(name_lst) == len(rev_lst)
    z = zip(name_lst, rev_lst)
    s = list(set(name_lst))
    s.sort()
    ls = []
    for name in s:
        ls.append((name, names_tuples(name ,z)))
    return ls

# This function does all function defined above at once 
def all_process():
    name_lst = get_beername_full()
    rev_lst = review_clean_full()
    all_result = word_group(name_lst, rev_lst)
    nest_lst = map(lambda x: x[1], all_result)
    name_lst = map(lambda x: x[0], all_result)
    pickle.dump(nest_lst, open('words_lst', 'wb'))
    pickle.dump(name_lst, open('name_lst', 'wb'))
    print "JOB DONE"









































