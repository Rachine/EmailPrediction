#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:08:51 2017

@author: paulinenicolas
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


def removing_stop_words(df_with_body) :
    
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

    data = []
    i=0
    for text in df_with_body['body'].tolist():
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
    data1 = pd.DataFrame({'word split': data})
    data11 = pd.concat([df_with_body, data1], axis=1, join='inner')
    del data11['body']
    return data11



def tfidf(df_split_word_train, df_split_word_test):
    corpus_train = df_split_word_train['word split'].tolist() 
    corpus_test = df_split_word_test['word split'].tolist() 
    t0 = time()
    vectorizer = TfidfVectorizer(min_df=1)
    X_train = vectorizer.fit_transform(corpus_train)
    X_test = vectorizer.transform(corpus_test)
    duration = time() - t0
    print("done in %fs" % (duration))
    print()
    return X_train, X_test



def closest_mail_test(df_test, df_train, X_train, X_test, number_keep, mid_send_recip):
    t0 = time()
    
    df_test['close_mids'] = [[]]*df_test.shape[0]
    df_test['close_mids_similarities'] = [[]]*df_test.shape[0]
    
    for idx in range(df_test.shape[0]):
        if idx%50 == 0:
            print(idx)
        x = X_test[idx]
        similarities = linear_kernel(x,X_train)[0]
        top_sim_idx = similarities.argsort()[-number_keep:][::-1]
        df_test['close_mids'][idx] = df_train['mid'][top_sim_idx].tolist()
        df_test['close_mids_similarities'][idx] = similarities[top_sim_idx]
        
    duration = time() - t0
    print("done in %fs" % (duration))
    print()
    df_test['recipients'] = 0
    
    t0 = time()
    for idx in range(df_test.shape[0]):
        if idx%10 == 0:
            print(idx)
        close_mids = df_test['close_mids'][idx]
        close_similarities = df_test['close_mids_similarities'][idx]
        receivers = {}
        for jdx,el in enumerate(close_mids):
            new_recs = mid_send_recip.loc[mid_send_recip['mid'] == str(el)]['recipient']
            new_recs = new_recs.tolist()
            for key_rec in new_recs:
                try:
                    receivers[key_rec] += close_similarities[jdx]
                except:
                    receivers[key_rec] = close_similarities[jdx]
        d = OrderedDict(sorted(receivers.items(), key=itemgetter(1)))
        df_test['recipients'][idx] = ' '.join(list(d.keys())[::-1][:10])
    
    
    duration = time() - t0
    print("done in %fs" % (duration))
    print()
    return df_test




 



