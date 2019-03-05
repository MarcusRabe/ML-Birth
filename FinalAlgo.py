#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:03:35 2019

@author: Marcus Rabe
@team: Team 2
"""

import pandas as pd
# prep
from sklearn.model_selection import train_test_split
import xgboost as xgb
# metrics
from sklearn import metrics

# import data
file = "birthweight_feature_set.xlsx"
birthweight = pd.read_excel(file)

# create a copy with dropped NAs
birthweight_dropped = birthweight.dropna()

# fill NAs and create the one needed marker column
birthweight['m_feduc'] = birthweight['feduc'].isnull().astype(int)
birthweight['meduc'] = birthweight['meduc'].fillna(birthweight_dropped['meduc'].mean().round(0))
birthweight['npvis'] = birthweight['npvis'].fillna(birthweight_dropped['npvis'].mean().round(0))
birthweight['feduc'] = birthweight['feduc'].fillna(birthweight_dropped['feduc'].mean().round(0))

# add new features
birthweight['bothBlck'] = (birthweight['mblck'] * birthweight['fblck'])
birthweight['bothOth'] = (birthweight['moth'] * birthweight['foth'])

# drop non needed features
birthweight = birthweight.drop(['omaps',
                                'fmaps',
                                'meduc',
                                'male',
                                'mwhte',
                                'mblck',
                                'moth',
                                'fwhte',
                                'fblck',], 1)

# outlier diictionary
OL_hi = {'npvis':18,
         'fage':50,
         'cigs':20,
         'drink':8
        }
OL_lo = {'npvis':4,
         'fage':18,
         'cigs':0,
         'drink':0
        }

# outlier flagging
def outlierflagging (df, OL_hi, OL_lo):
    for key in OL_hi.keys():
        df['out_'+key] = 0
        df['out_'+key] = df[key].apply(lambda val: 1 if val > OL_hi[key] else -1 if val < OL_lo[key] else 0)
    return df

birthweight = outlierflagging(birthweight, OL_hi, OL_lo)

# attributes
X = birthweight.drop(['bwght'], 1)
# lables
y = birthweight['bwght']

# create the split
X_train_n, X_test_n, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=508)

# train the Algorithm
xg_reg2 = xgb.XGBRegressor(objective ='reg:linear',
                          colsample_bytree = 1,
                          learning_rate = 0.35,
                          max_depth = 2,
                          alpha = 0.01,
                          n_estimators = 9,
                          subsample = 1)
xg_reg2.fit(X_train_n,
            y_train)

# Score the algorithm
score = xg_reg2.score(X_test_n,
                      y_test)
train_score = xg_reg2.score(X_train_n,
                            y_train)
diff = train_score - score

print("Testing score: %f" % (score))
print("Training score: %f" % (train_score))
print("Diff: %f" % (diff))