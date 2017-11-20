#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:32:58 2017

@author: vishwas
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import pairwise as pw

def cf_prerprocess(ratings_mat):
    g_avg = np.mean(ratings_mat)
    user_bias = np.sum(ratings_mat, axis =1)/ratings_mat.shape[1]
    ratings_mat = ratings_mat + g_avg
    ratings_mat = ratings_mat - np.expand_dims(user_bias,1)
    return ratings_mat, g_avg

def user_cf_proc(user_inp, ratings_mat, beer_names):
    beer_name_input = [i[0] for i in user_inp]
    ratings_input = [i[0] for i in user_inp]
    user_data = np.repeat(np.nan, ratings_mat.shape[1])
    beer_idx = [beer_names.keys()[beer_names.values().index(i)] for i in beer_name_input]
    for i, j in enumerate(beer_idx):
        user_data[j] = ratings_input[i]

    return user_data

def cf_rec(user_data, ratings, g_avg, beer_names, neighbors=10, num_recs=5):

    '''Find indices of observed and missing values'''
    index = np.where(~np.isnan(user_data))[0]
    missing_index = np.where(np.isnan(user_data))[0]

    '''take items as columns'''
    ratings_mat_red = ratings[:, index]
    ratings_mat_miss = ratings[:, missing_index]

    user_data_new = user_data[index]

    '''normalize via global avg mu'''
    user_data_new = user_data_new - user_data_new.mean() + global_avg

    '''compute euclidean distance and cosine similarity'''
    pw_cos = pw.cosine_similarity(ratings_mat_red,user_data_new).flatten()

    # largest values are most similar users
    pw_cos_df = pd.DataFrame([pw_cos]).transpose()
    cos_topn = pw_cos_df.nlargest(neighbors,0)

    # turn distance/similarity into weights
    # might need to inspect distance again, right now I simply reversed the weights
    cos_weights = np.matrix(cos_topn / sum(cos_topn[0]))

    # get ratings from the top n users for the missing beers
    cos_miss_ratings = ratings_mat_miss[cos_topn.index,:]

    # weigh ratings
    cos_new_ratings = pd.DataFrame(cos_miss_ratings.transpose() * cos_weights)

    # find top rating(s)
    cos_new_index = cos_new_ratings.nlargest(num_recs, 0).index[:]

    beer_ind_cos = missing_index[cos_new_index]

    # match index with beer name
    return [beer_names[ind].decode("utf-8", "ignore").encode("ascii", "ignore") for ind in beer_ind_cos]

if __name__ == '__main__':

    # load matrix here
    ratings_mat = np.load('ratings_mat.npy')

    # load beer dictionary here
    with open('beer_dict.pickle', 'r') as f:
        beer_names = pickle.load(f)

    test_users = [('Yazoo Embrace the Funk Series: Deux Rouges', 20), ("Iron Hill Bourbon Porter", 3)]

    rating_mat, global_avg = cf_prerprocess(ratings_mat)
    user_data = user_cf_proc(test_users, ratings_mat, beer_names)
    print beer_names
    print cf_rec(user_data, ratings_mat, global_avg, beer_names)



