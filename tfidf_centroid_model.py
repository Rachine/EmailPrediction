#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:07:13 2017

@author: paulinenicolas
"""


import pandas as pd
import os
import numpy as np
from function import *
import datetime 
from sklearn.feature_extraction.text import TfidfVectorizer


init_path = '/Users/paulinenicolas/EmailPrediction'
path_to_results = init_path + '/results'

print('Loading the 4 intial datasets (train and test)..')
#Reading the two datasets and transform them into pandas df
train_info = pd.read_csv(init_path + '/data/training_info.csv' , sep = ',')
#train_set0 = pd.read_csv(init_path + '/data/training_set.csv' , sep = ',')
train_set = pd.read_csv(init_path + '/data/training_set.csv' , sep = ',')

train_info = train_info[np.logical_or(train_info['date'].str.contains('199'),train_info['date'].str.contains('200'))].reset_index()
mids_train = train_info['mid'].tolist()

i=0
        
for sender in train_set['sender']:
    lis = [int(w) for w in train_set.loc[train_set['sender'] == sender]['mids'].tolist()[0].split() if int(w) in mids_train]
    train_set.loc[train_set['sender'] == sender]['mids']= ' '.join( [str(w) for w in lis])
    i+=1
    if i%10==0:
        print(i)

train_info['date']= pd.to_datetime(train_info['date'])
train_info = train_info.loc[train_info['date']>(np.max(train_info['date']) - datetime.timedelta(6*365/12))].reset_index()

mids_train = train_info['mid'].tolist()

i=0
        
for sender in train_set['sender']:
    lis = [int(w) for w in train_set.loc[train_set['sender'] == sender]['mids'].tolist()[0].split() if int(w) in mids_train]
    train_set.loc[train_set['sender'] == sender]['mids']= ' '.join( [str(w) for w in lis])
    i+=1
    if i%10==0:
        print(i)


#train_info = train_info[train_info.shape[0]-25000:].reset_index()
#idx = np.arange(train_info.shape[0])
#np.random.shuffle(idx)
#
#train_idx = idx[:10000].tolist()
#
#train_info = df_info.iloc[train_idx,:].reset_index()
#
#
#train_set = train_set0.copy()
#mids_train = train_info['mid'].tolist()
#i=0
#for sender in train_set0['sender']:
#    lis = [int(w)  for w in train_set.loc[train_set['sender'] == sender]['mids'].tolist()[0].split() if int(w) in mids_train]
#    train_set['mids'][i] = ' '.join([str(w) for w in lis])
#    i+=1

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

#data_split_word_test['mid'] = data_split_word_test['mid'].astype(str)
#data_split_word_train['mid'] = data_split_word_train['mid'].astype(str)





i = 0
for sender in test_set['sender']:
    
    print('sender #', i)
    
    list_mid_train = train_set.loc[train_set['sender'] == sender]['mids'].str.split().tolist()[0]
    list_mid_test = test_set.loc[test_set['sender'] == sender]['mids'].str.split().tolist()[0]
   
    train_info_sender = data_split_word_train.loc[data_split_word_train['mid'].isin([int(i) for i in list_mid_train])].reset_index(drop=True)
        
    test_info_sender = data_split_word_test.loc[data_split_word_test['mid'].isin([int(i) for i in list_mid_test])].reset_index(drop=True)
    mid_send_recip_train_sender  = mid_send_recip_train.loc[mid_send_recip_train['sender'] == sender].reset_index(drop=True)
    test_set_sender = test_set.loc[test_set['sender']== sender]
    train_info_sender['mid'] = train_info_sender['mid'].astype(str)
    test_info_sender['mid'] = test_info_sender['mid'].astype(str)
    
    if sender == 'alex@pira.com':
        prediction_test = []
        for mid in list_mid_test:
            prediction_test.append((mid,'alex@pira.com'))
    else:
        model = tfidf_centroid()
        model.fit(train_info_sender, mid_send_recip_train_sender)
        prediction_test = model.predict(test_set_sender, test_info_sender, 10, sender)
        
    df_result = pd.DataFrame(prediction_test, columns=['mid', 'recipients'])
    if i==0:
        result_final = df_result
    else:
        result_final = pd.concat([result_final,df_result])
        
    i+=1
    

    
result_final.to_csv('tf_idf_result_new_6_month_sqrt.csv', sep=',',index=False)
    
    
    




