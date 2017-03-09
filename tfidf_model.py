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
recipient_info_train['mid'] = recipient_info_train['mid'].astype(str)

mid_send_recip_train  = sender_info_train.merge(recipient_info_train, how='inner', left_on='mid', right_on='mid', )

# For each document in the dataset, do the preprocessing for removing stop words
data_split_word_train = removing_stop_words(train_info)
print('Stop Word removed for training..')
data_split_word_test = removing_stop_words(test_info)
print('Stop Word removed for testing..')

#adress book for all sender/recipients
address_book_train = mid_send_recip_train.groupby(['sender', 'recipient'])['mid'].apply(list).reset_index()

dict_prob_r = {}

for index, row in address_book_train.iterrows():
    if  row[1] in dict_prob_r.keys():
        dict_prob_r[row[1]] += len(row[2])/train_info.shape[0]
    else:
        dict_prob_r[row[1]] = len(row[2])/train_info.shape[0]

predict_test = []

corpus_train = data_split_word_train['word split'].tolist()
vectorizer = TfidfVectorizer(min_df=1)
X_train = vectorizer.fit_transform(corpus_train)



i=0
for sender in test_set['sender']:
    
    print(i)
    t0 = time()
    
    model = tfidf_centroid()
    model.fit(data_split_word_train, train_set, address_book_train, X_train,sender)
    
    duration = time() - t0
    print("training phase done in %fs" % (duration))
    print()
    
    list_mid =  test_set.loc[test_set['sender'] == sender]['mids'].str.split().tolist()[0]
    t0 = time()
    for mid in list_mid:
        prediction = model.predict(mid, sender, test_set, data_split_word_test, dict_prob_r, vectorizer,  10)
        predict_test.append((mid, prediction))
   
    duration = time() - t0
    print("testing phase done in %fs" % (duration))
    print()
    
    i+=1


df_result = pd.DataFrame(predict_test, columns=['mid', 'recipients'])
df_result.to_csv('tf_idf_result.csv', sep=',',index=False)