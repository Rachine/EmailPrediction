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


init_path = os.getcwd()
pathname_train_info = init_path + '/data/training_info.csv'
pathname_train_set = init_path + '/data/training_set.csv'


print('Merging the initial 2 datasets..')
#Reading the two datasets and transform them into pandas df
df_info = pd.read_csv(pathname_train_info , sep = ',')
df_set = pd.read_csv(pathname_train_set , sep = ',')

#Seperating the different mail id/recipients id
sender_info = pd.concat([pd.Series(row['sender'], row['mids'].split(' '))              
                    for _, row in df_set.iterrows()]).reset_index()
recipient_info = pd.concat([pd.Series(row['mid'], row['recipients'].split(' '))              
                    for _, row in df_info.iterrows()]).reset_index()
#renaming columns
sender_info.columns = ['mid', 'sender']
recipient_info.columns = ['recipient', 'mid']
#Changing type mid  into string for the following merge
recipient_info['mid'] = recipient_info['mid'].astype(str)
#Merging two dataset : final granularity : sender mail - recipient mail - number of mail sent from sendr to recipient
merge_info  = sender_info.merge(recipient_info, how='inner', left_on='mid', right_on='mid')
send_recip_nb_mail = merge_info.groupby(['sender', 'recipient'], as_index=False).count()




print('New pandas Dataset created called send_recip_nb_mail')
print('----------------------------------------------------')
print('  ')


stopwords = ['me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                 'than', 'too', 'very', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z',
                 'can', 'will', 'just', 'don', 'should', 'now', ' ']


# For each document in the dataset, do the preprocessing

data = []
i=0
for text in df_info['body'].tolist():
    #    
    # Remove punctuation 
    punctuation = set(string.punctuation)
    doc = ''.join(w for w in text.lower() if w not in punctuation)
    # Stopword removal
    doc = [w for w in doc.split() if w not in stopwords]
    doc = [w for w in doc if not (any(c.isdigit() for c in w))]
     # Stemming
    stemmer=PorterStemmer()
    doc2= [stemmer.stem(w) for w in doc]
    # Covenrt list of words to one string
    doc2 = ' '.join(doc2)
    i+=1
    if i%1000==0:
        print(i)
    data.append(doc2)   # list data contains the preprocessed documents


print('Stop Word removed..')

df1 = pd.DataFrame({'word split': data})
df_word = pd.concat([df_info, df1,], axis=1, join='inner')
del df_word['body']

corpus = df_word['word split'].tolist()
vectorizer = TfidfVectorizer(min_df=1)

vectorizer.fit_transform(corpus)


