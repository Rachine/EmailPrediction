#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:31:55 2017

@authors : Pauline Nicolas Leo Treguer Riad Rachid
"""

import numpy as np
import pandas as pd
import os
import nltk
from nltk.stem.porter import *
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from time import time
from collections import OrderedDict
from operator import itemgetter
from function import removing_stop_words, tfidf, closest_mail_test

init_path = os.getcwd()
path_to_results = init_path + '/results'

print('Loading the 4 intial datasets (train and test)..')
#Reading the two datasets and transform them into pandas df
df_info = pd.read_csv(init_path + '/data/training_info.csv' , sep = ',')
df_set = pd.read_csv(init_path + '/data/training_set.csv' , sep = ',')

test_info = pd.read_csv(init_path +'/data/test_info.csv',  sep=',', header=0)
test_set = pd.read_csv(init_path + '/data/test_set.csv', sep=',', header=0)

##A BIT OF PRE-PROCESSING

#Seperating the different mail id/recipients id
sender_info = pd.concat([pd.Series(row['sender'], row['mids'].split(' '))              
                     for _, row in df_set.iterrows()]).reset_index()
recipient_info = pd.concat([pd.Series(row['mid'], row['recipients'].split(' '))              
                     for _, row in df_info.iterrows()]).reset_index()#renaming columns
sender_info.columns = ['mid', 'sender']
recipient_info.columns = ['recipient', 'mid']
#Changing type mid  into string for the following merge
recipient_info['mid'] = recipient_info['mid'].astype(str)
#Merging two dataset : final granularity : sender mail - recipient mail - number of mail sent from sendr to recipient
mid_send_recip  = sender_info.merge(recipient_info, how='inner', left_on='mid', right_on='mid')
send_recip_count_mail = mid_send_recip.groupby(['sender', 'recipient'], as_index=False).count()
 
 
print('New pandas Dataset created called send_recip_nb_mail')

# For each document in the dataset, do the preprocessing  
data_split_word_train = removing_stop_words(df_info)
print('Stop Word removed for training..')
data_split_word_test = removing_stop_words(test_info)
print('Stop Word removed for testing..')


print(" X_train : Extracting features from the train data using tf-idf")
print("X_test : Extracting features from the test data using the same vectorizer")
X_train, X_test = tfidf(data_split_word_train, data_split_word_test)



print("Compute closest for every email in the dataset test")
final_df_test = closest_mail(data_split_word_test, data_split_word_train, X_train, X_test, 30, mid_send_recip)
df_word_test_final = df_test[['mid','recipients']]

df_word_test_final.to_csv(path_to_results+'/tf_idf_result.csv', sep=',',index=False)
