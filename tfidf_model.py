#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:31:55 2017

@authors : Pauline Nicolas Leo Treguer Riad Rachid
"""

import pandas as pd
import os
from function import *

init_path = os.getcwd()
path_to_results = init_path + '/results'

print('Loading the 4 intial datasets (train and test)..')
#Reading the two datasets and transform them into pandas df
train_info = pd.read_csv(init_path + '/data/training_info.csv' , sep = ',')
train_set = pd.read_csv(init_path + '/data/training_set.csv' , sep = ',')

test_info = pd.read_csv(init_path +'/data/test_info.csv',  sep=',', header=0)
test_set = pd.read_csv(init_path + '/data/test_set.csv', sep=',', header=0)


#Seperating the different mail id/recipients id
sender_info_train = pd.concat([pd.Series(row['sender'], row['mids'].split(' '))              
                     for _, row in train_set.iterrows()]).reset_index()

recipient_info_train = pd.concat([pd.Series(row['mid'], row['recipients'].split(' '))              
                     for _, row in train_info.iterrows()]).reset_index()#renaming columns
sender_info_train.columns = ['mid', 'sender']

recipient_info_train.columns = ['recipient', 'mid']
#Changing type mid  into string for the following merge
recipient_info_train['mid'] = recipient_info_train['mid'].astype(str)
#Merging two dataset : final granularity : sender mail - recipient mail - number of mail sent from sendr to recipient
mid_send_recip_train  = sender_info_train.merge(recipient_info_train, how='inner', left_on='mid', right_on='mid')
send_recip_count_mail_train = mid_send_recip_train.groupby(['sender', 'recipient'], as_index=False).count()

##A BIT OF PRE-PROCESSING
#train_set['mids'] = train_set['mids'].str.split(' ')
#test_set['mids'] = test_set['mids'].str.split(' ')

# For each document in the dataset, do the preprocessing for removing stop words
data_split_word_train = removing_stop_words(train_info)
print('Stop Word removed for training..')
data_split_word_test = removing_stop_words(test_info)
print('Stop Word removed for testing..')

model = tfidf_centroid()
model.fit(data_split_word_train, mid_send_recip_train)
prediction_test = model.predict(test_set, data_split_word_test, 10)

df_result = pd.DataFrame(prediction_test, columns=['recipients'])
result_final = pd.concat([test_info['mid'],df_result], axis=1)
result_final.to_csv('tf_idf_result.csv', sep=',',index=False)