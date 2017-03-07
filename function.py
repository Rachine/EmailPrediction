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
import pdb


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



def closest_mail(df_test, df_train, X_train, X_test, number_keep):
    t0 = time()
    
    df_train['recipients'] = df_train['recipients'].str.split(' ')
    
    df_test['close_mids'] = [[]]*df_test.shape[0]
    df_test['close_mids_similarities'] = [[]]*df_test.shape[0]
    
    
    for idx in range(df_test.shape[0]):
        if idx%50 == 0:
            print(idx)
        x = X_test[idx]
        similarities = linear_kernel(x,X_train)[0]
        top_sim_idx = similarities.argsort()[-number_keep:][::-1]
        #df_test['close_mids'][idx] = df_train['mid'][top_sim_idx].tolist()
        #df_test['close_mids_similarities'][idx] = similarities[top_sim_idx]
        
        close_mids = df_train['mid'][top_sim_idx].tolist()
        close_similarities = similarities[top_sim_idx]
        receivers = {}
        for jdx,el in enumerate(close_mids):
            new_recs = df_train.loc[df_train['mid'] == el]['recipients']
            new_recs = new_recs.tolist()[0]
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
    df_test['recipients'] = 0
    
    return df_test



class tfidf_centroid():
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(min_df=1)

    def fit(self, df_split_word_train, mid_send_recip_train):
        
        corpus_train = df_split_word_train['word split'].tolist() 
        t0 = time()
        self.X_train = self.vectorizer.fit_transform(corpus_train)
        
        #computation of the address book
        address_book_train = mid_send_recip_train.groupby(['sender', 'recipient'])['mid'].apply(list).reset_index()
        
        self.dict_centroid={}
        
        for index, row in address_book_train.iterrows():
            
            if index%1000==0:
                print (index)
            
            #Compute tf-idf centroid for each couple (sender, receiver) and fill the dictionnary
            if row[0] in self.dict_centroid.keys():
                list_idx = df_split_word_train[df_split_word_train['mid'].isin([int(i) for i in row[2]])].index.tolist()
                self.dict_centroid[row[0]][row[1]]= self.X_train[list_idx].sum(axis=0)
            else:
                self.dict_centroid[row[0]] = {}
                list_idx = df_split_word_train[df_split_word_train['mid'].isin([int(i) for i in row[2]])].index.tolist()
                self.dict_centroid[row[0]][row[1]]= self.X_train[list_idx].sum(axis=0)
                
        self.prob_r = mid_send_recip_train.groupby(['recipient'])['mid'].count()/df_split_word_train.shape[0]
        self.prob_r = self.prob_r.reset_index()
        self.prob_s_given_r = mid_send_recip_train.groupby(['recipient', 'sender'])['mid'].count().reset_index() 
        self.prob_s_given_r = self.prob_s_given_r.merge(mid_send_recip_train.groupby(['recipient'])['mid'].count().reset_index(), how='inner', left_on='recipient', right_on='recipient')
        self.prob_s_given_r['prob_s_r'] =  self.prob_s_given_r['mid_x']/self.prob_s_given_r['mid_y']
        self.prob_s_given_r = self.prob_s_given_r[['recipient', 'sender', 'prob_s_r']]
        duration = time() - t0
        print("done in %fs" % (duration))
        print()

    def predict(self, test_set, df_split_word_test, number_keep):
        sender_info_test = pd.concat([pd.Series(row['sender'], row['mids'].split(' '))              
                     for _, row in test_set.iterrows()]).reset_index()
        
        sender_info_test.columns = ['mid', 'sender']
        df_split_word_test['mid'] = df_split_word_test['mid'].astype(str)
        df_split_word_test_bis = df_split_word_test.merge(sender_info_test[['mid', 'sender']], how='inner', left_on='mid', right_on='mid')
        
        corpus_test = df_split_word_test['word split'].tolist() 
        X_test = self.vectorizer.transform(corpus_test)
        
        predict_test = []
        
        for idx in range(X_test.shape[0]):
            sender = df_split_word_test_bis['sender'][idx]
            cosine_list = []
            prob_list = [] 
            tot_cosine = 0
            for recip in self.dict_centroid[sender]:
                cosine = linear_kernel(X_test[idx], self.dict_centroid[sender][recip])
                tot_cosine += cosine
                
                prob_r_select = float(self.prob_r.loc[self.prob_r['recipient'] == recip]['mid'])
                prob_s_r = float(self.prob_s_given_r.loc[(self.prob_s_given_r['sender'] == sender) & (self.prob_s_given_r['recipient'] == recip) ]['prob_s_r'])
                
                cosine_list.append((recip,float(cosine)))
                prob_list.append((recip, prob_r_select, prob_s_r))
 
            
            #we take the ten biggest cosine similarity for each given mail-sender
            maximum = max(cosine_list,key=itemgetter(1))[1] 
            minimum =min(cosine_list,key=itemgetter(1))[1]  
            cosine_list = [(cosine[0], (cosine[1]- minimum)/(maximum-minimum)) if maximum!=minimum else (cosine[0],1) for cosine in cosine_list]
            
            final_prob = []
            for i in range(len(cosine_list)):
                final_prob.append((cosine_list[i][0], cosine_list[i][1]*prob_list[i][1]*prob_list[i][2]))
            
            final_prob = sorted(final_prob, key=lambda prob: prob[1], reverse=True)[:number_keep]
            
            predict_test.append(' '.join([x[0] for x in final_prob]))
            
            if idx%100 == 0:
                print(idx)
            
        return predict_test
        
        
        
        


 




