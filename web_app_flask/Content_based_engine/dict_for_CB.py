#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:31:04 2017

@author: vishwas
"""

import pandas as pd
import numpy as np

df1 = pd.read_csv("raw_data/Beer_review_data/beerreview1_13.csv")
df2 = pd.read_csv("raw_data/Beer_review_data/beerreview14_26.csv")
df3 = pd.read_csv("raw_data/Beer_review_data/beerreview27_39.csv")
df4 = pd.read_csv("raw_data/Beer_review_data/beerreview40_51.csv")

df = pd.concat([df1,df2,df3,df4], axis = 0, ignore_index = True)

import pickle
import re

df = df[['user_name', 'user_info', 'state', 'beer_name', 'overall', 'user_rating', 'aroma',
        'appearance', 'palate', 'taste']]


df[df['user_name'].isnull()].index.tolist()   # locate the user with missing name

un = df['user_name']
print len(un)
clean_name = lambda x: ''.join(e for e in re.sub('\([^)]*\)', '', x) if e.isalnum())
un2 = map(lambda x: clean_name(x), un)  # clean user_name col
print len(un2)

df['user_name'] = un2

ui = df['user_info']
ui[:5]
find_date = lambda x: re.search('[A-Z]{3} \d{1,2}, \d{4}$', x)
user_date = map(lambda x: find_date(x).group() if find_date(x) else np.nan, ui)
user_date.count(np.nan)   # count missing values after extracting date
pd.Series(map(lambda x: len(x), [item for item in user_date if item is not np.nan])).unique()
user_date = pd.to_datetime(user_date, errors='ignore', format='%b %d, %Y')
df['user_date'] = user_date
df = df[['user_name', 'user_date', 'state', 'beer_name', 'overall', 'taste', 'aroma', 'appearance', 'palate', 'user_rating']]

st = df['state']
len(st.unique())   # state variable is very clean

bn = df['beer_name']
bn = map(lambda x: x.decode('utf-8', 'ignore').encode('ascii', 'ignore'), bn)
bn = map(lambda x: re.sub('\s+', ' ', x).strip(), bn)
df['beer_name'] = bn

df_1 = df.groupby('beer_name').agg({'overall': 'mean', 'user_rating': 'mean'})

dict = {}
for i in range(1269):
        dict[df2.index[i]] = [round(df2.iloc[i, 0], 1)*4, round(df2.iloc[i, 1], 1)]
pickle.dump(dict, open('dict_for_CB_table.p', 'wb'))







