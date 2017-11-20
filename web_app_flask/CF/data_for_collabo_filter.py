#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:28:01 2017

@author: vishwas
"""
import pandas as pd
import pickle


def get_dict(x):
    '''create dictionary of users and beers'''
    if x == 'user':
        dict = {}
        key_lst = df_tf['user'].tolist()
        value_lst = beer_ratings['user_name'].tolist()
        for i in range(len(key_lst)):
            dict[key_lst[i]] = value_lst[i]
        return dict
    if x == 'beer':
        dict = {}
        key_lst = df_tf['beer'].tolist()
        value_lst = beer_ratings['beer_name'].tolist()
        for i in range(len(key_lst)):
            dict[key_lst[i]] = value_lst[i]
        return dict


if __name__ == '__main__':

    # load the dataframe with three needed columns
    beer_ratings = pd.read_csv("beer_ratings.csv")
    # factorize user_name and beer_name columns and create a new dataframe
    df_tf = pd.DataFrame()
    df_tf['user'] = pd.factorize(beer_ratings['user_name'])[0]
    df_tf['beer'] = pd.factorize(beer_ratings['beer_name'])[0]
    df_tf['rating'] = beer_ratings['user_rating']

    # generate two dictionaries for user and beer
    user_dict = get_dict('user')
    beer_dict = get_dict('beer')
    print 'two dictionaries are generated'

    # export three objects
    pickle.dump(user_dict, open('user_dict.pickle', 'wb'))
    pickle.dump(beer_dict, open('beer_dict.pickle', 'wb'))
    df_tf.to_csv('user_beer_rating_facorized.csv', index=False)
    print 'objects are exported'






