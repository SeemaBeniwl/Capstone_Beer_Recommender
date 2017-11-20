#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:00:32 2017

@author: vishwas
"""
import numpy as np
import pandas as pd

df = pd.read_csv("user_beer_rating_facorized.csv")
df.dtypes
n_users  = df.user.unique().shape[0]
n_items = df.beer.unique().shape[0]
'''computing the implicit matrix of the data'''
implicit_mat = df.pivot(index = 'user',
                        columns = 'beer',
                        values = 'rating').notnull().as_matrix().astype(float)


data_matrix = np.zeros((n_users, n_items))
for line in df.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]
np.save('ratings_mat', data_matrix)




from sklearn.cross_validation import train_test_split
train_data, test_data = train_test_split(df, test_size = 0.25)
'''memory based collaborative filter'''
'''creating the matrix for the cosine similarity'''

train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
'''prediction function'''
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
item_prediction = predict(train_data_matrix, item_similarity, type='item')

user_prediction = predict(train_data_matrix, user_similarity, type='user')
''''Getting the Accuracy of the prediction'''
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))



'''model based collaborative filtering'''

import scipy.sparse as sp
from scipy.sparse.linalg import svds

#get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print 'User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix))



'''Using different package to do the svd for collaborative filter'''

from surprise import SVD
from surprise import evaluate
import os
import sys
from surprise import Reader
from surprise import Dataset
file_path = os.path.expanduser('user_rating_fac.csv')
reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader) 






algo = SVD()







