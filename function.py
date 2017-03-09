#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:40:02 2017

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
import pdb
from scipy.sparse.linalg import norm
import scipy

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
    data_result = pd.DataFrame({'word split': data})
    data_result = pd.concat([df_with_body, data_result], axis=1, join='inner')
    del data_result['body']
    return data_result




class tfidf_centroid():
    
    def __init__(self):
        pass

    def fit(self, df_split_word_train, train_set, address_book_train, X_train, sender):
        
        address_book_train_sender = address_book_train.loc[address_book_train['sender']==sender]
        
        #computation of the address boo
        #'becky.spencer@enron.com'
        self.dict_centroid={}
        self.dict_prob_r_s = {}
        
        for index, row in address_book_train_sender.iterrows():
                
            list_idx = df_split_word_train[df_split_word_train['mid'].isin([int(i) for i in row[2]])].index.tolist()
                                
            #filling dictionnary of prob_s_r
            self.dict_prob_r_s[row[1]] = len(list_idx)/address_book_train.loc[address_book_train['recipient']==row[1]]['mid'].apply(len).sum()
            
            #Compute tf-idf centroid for each couple (sender, receiver) and fill the dictionnary
            self.dict_centroid[row[1]] = X_train[list_idx].sum(axis=0)


    def predict(self, mid, sender, test_set, df_split_word_test, dict_prob_r, vectorizer, number_keep):

        df_split_word_test['mid'] = df_split_word_test['mid'].astype(str)
        df_split_word_test_bis = df_split_word_test.loc[df_split_word_test['mid'] == mid]
        
        corpus_test = df_split_word_test_bis['word split'].tolist() 
        X_test = vectorizer.transform(corpus_test)

        cosine_list_r_s = []
        prob_list = [] 
        
        i=0
        for recip in self.dict_centroid:

            prob_r = dict_prob_r[recip]
            prob_r_s = self.dict_prob_r_s[recip]
                
            cosine_r_s = linear_kernel(X_test, self.dict_centroid[recip])/norm(scipy.sparse.csr_matrix(self.dict_centroid[recip]))
            cosine_list_r_s.append((recip,float(cosine_r_s)))

            prob_list.append((recip, prob_r_s, prob_r))
            
            #we take the ten biggest cosine similarity for each given mail-sender
            maximum_r_s = max(cosine_list_r_s,key=itemgetter(1))[1] 
            minimum_r_s =min(cosine_list_r_s,key=itemgetter(1))[1] 
            
            cosine_list_r_s = [(cosine[0], (cosine[1]- minimum_r_s)/(maximum_r_s - minimum_r_s)) if maximum_r_s!=minimum_r_s else (cosine[0],1) for cosine in cosine_list_r_s]
            final_prob = []
            
            for i in range(len(cosine_list_r_s)):
                final_prob.append((cosine_list_r_s[i][0], (cosine_list_r_s[i][1])*prob_list[i][1]*prob_list[i][2]))
            
            final_prob = sorted(final_prob, key=lambda prob: prob[1], reverse=True)[:number_keep]
            
            final_prob = ' '.join([x[0] for x in final_prob])

            
        return final_prob



#def tfidf(df_split_word_train, df_split_word_test):
#    corpus_train = df_split_word_train['word split'].tolist() 
#    corpus_test = df_split_word_test['word split'].tolist() 
#    t0 = time()
#    vectorizer = TfidfVectorizer(min_df=1)
#    X_train = vectorizer.fit_transform(corpus_train)
#    X_test = vectorizer.transform(corpus_test)
#    duration = time() - t0
#    print("done in %fs" % (duration))
#    print()
#    return X_train, X_test
#
#
#
#def closest_mail(df_test, df_train, X_train, X_test, number_keep):
#    t0 = time()
#    
#    df_train['recipients'] = df_train['recipients'].str.split(' ')
#    
#    df_test['close_mids'] = [[]]*df_test.shape[0]
#    df_test['close_mids_similarities'] = [[]]*df_test.shape[0]
#    
#    
#    for idx in range(df_test.shape[0]):
#        if idx%50 == 0:
#            print(idx)
#        x = X_test[idx]
#        similarities = linear_kernel(x,X_train)[0]
#        top_sim_idx = similarities.argsort()[-number_keep:][::-1]
#        #df_test['close_mids'][idx] = df_train['mid'][top_sim_idx].tolist()
#        #df_test['close_mids_similarities'][idx] = similarities[top_sim_idx]
#        
#        close_mids = df_train['mid'][top_sim_idx].tolist()
#        close_similarities = similarities[top_sim_idx]
#        receivers = {}
#        for jdx,el in enumerate(close_mids):
#            new_recs = df_train.loc[df_train['mid'] == el]['recipients']
#            new_recs = new_recs.tolist()[0]
#            for key_rec in new_recs:
#                try:
#                    receivers[key_rec] += close_similarities[jdx]
#                except:
#                    receivers[key_rec] = close_similarities[jdx]
#        d = OrderedDict(sorted(receivers.items(), key=itemgetter(1)))
#        df_test['recipients'][idx] = ' '.join(list(d.keys())[::-1][:10])
#
#        
#    duration = time() - t0
#    print("done in %fs" % (duration))
#    print()
#    df_test['recipients'] = 0
#    
#    return df_test