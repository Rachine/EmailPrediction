#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:29:19 2017

@authors : Pauline Nicolas Leo Treguer Riad Rachid
"""

import numpy as np
import pandas as pd
import os
import nltk
from nltk.stem.porter import *
import string

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




#Calculation of P(R) # emails received by recipient R/ total # emails sent at that point in time
## First method using frequency of emails

def Prob_r(send_recip_nb_mail, df_info):
    prob_R = send_recip_nb_mail[['recipient', 'mid']].groupby(['recipient'], as_index=False).sum()
    prob_R['prob_R'] = prob_R['mid']/df_info.shape[0]
    return prob_R

Prob_R = Prob_r(send_recip_nb_mail, df_info)



print('Prob_R, dataframe containing probability of each recipient')
print('----------------------------------------------------------')
print(Prob_R.head())
print('----------------------------------------------------------')
print('  ')




#Got it in a dictionnary form
#dict_prob_R = dict([(recip,mid) for recip, mid in zip(prob_R.recipient, prob_R.mid)])


# Calculation of P(S|R) 
## First method Pfreq(S|R) = # email from S to R / sum_on_s(#email for s to R)

def Prob_s_sachant_r_freq(send_recip_nb_mail, prob_R):
    prob_S_sachant_R = send_recip_nb_mail.merge(prob_R[['recipient', 'mid']], how='inner', left_on = 'recipient', right_on = 'recipient')
    prob_S_sachant_R['prob_S_R'] =  prob_S_sachant_R['mid_x']/prob_S_sachant_R['mid_y']
    prob_S_sachant_R = prob_S_sachant_R[['sender', 'recipient', 'prob_S_R']]
    return prob_S_sachant_R

Prob_S_sachant_R_freq = Prob_s_sachant_r_freq(send_recip_nb_mail, Prob_R)

print('Prob_S_sachant_R_freq, dataframe containing probability of each sender given recipient')
print('--------------------------------------------------------------------------------------')
print(Prob_S_sachant_R_freq.head())
print('-------------------------------------------------------------')
print(' ')



# Calculation of P(E|R,S)

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


#Granularity one recipient by line
print('Splitting recipients : one row per recipient per email')  
s = df_word['recipients'].str.split(' ').apply(pd.Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'recipient'
del df_word['recipients']
df_word2 = df_word.join(s)
#Granularity one word by line
print('Splitting words : one row per word per recipient')  
t = df_word['word split'].str.split(' ').apply(pd.Series, 1).stack()
t.index = t.index.droplevel(-1)
t.name = 'word'
del df_word2['word split']
df_word3 = df_word2.join(t)



#Probability of each word given sender and recipient
df_word3['mid'] = df_word3['mid'].astype(str)
df_word4 = df_word3.merge(sender_info, how='inner', left_on=['mid'], right_on=['mid'])

sender_recip_nb_word = df_word4.groupby(['sender', 'recipient'], as_index=False).count()
sender_recip_nb_word = sender_recip_nb_word.rename(columns={'word': 'tot_word_sender_recip'})

df_word5 = df_word4.groupby(['sender', 'recipient', 'word'], as_index=False).count()
df_word5 = df_word5.rename(columns={'mid': 'word_sender_recip'})
prob_w_sachant_S_R = df_word5.merge(sender_recip_nb_word, how='inner', left_on=['recipient', 'sender'], right_on=['recipient', 'sender'])
prob_w_sachant_S_R['prob_w_sachant_S_R'] = prob_w_sachant_S_R['word_sender_recip']/prob_w_sachant_S_R['tot_word_sender_recip']
del prob_w_sachant_S_R['date_x']
del prob_w_sachant_S_R['date_y']
del prob_w_sachant_S_R['mid']
del prob_w_sachant_S_R['tot_word_sender_recip']

#Probability of each word given sender and recipient

recip_nb_word = sender_recip_nb_word.groupby(['recipient'], as_index=False).sum()
recip_nb_word = recip_nb_word.rename(columns={'tot_word_sender_recip': 'tot_word_recip'})

df_word6 = df_word5.groupby(['recipient', 'word'], as_index=False).sum()
df_word6 = df_word6.rename(columns={'tot_mail_sender_recip_word': 'word_sender_recip'})
prob_w_sachant_R = df_word6.merge(recip_nb_word, how='inner', left_on=['recipient'], right_on=['recipient'])
prob_w_sachant_R['prob_w_sachant_R'] = prob_w_sachant_R['word_sender_recip']/prob_w_sachant_R['tot_word_recip']
del prob_w_sachant_R['date_x']
del prob_w_sachant_R['date_y']
del prob_w_sachant_R['mid']
del prob_w_sachant_R['tot_word_recip']
del prob_w_sachant_R['word_sender_recip'] 


#Probability of each word given sender and recipient
nb_word = len(t)

prob_w = df_word6.groupby(['word'], as_index=False).sum()
prob_w = prob_w.rename(columns={'word_sender_recip': 'word_oc'})
prob_w['prob_w']= prob_w['word_oc']/nb_word
del prob_w['word_oc']
del prob_w['date']


 
prob = prob_w.merge(prob_w_sachant_R, how='inner', left_on=['word'], right_on=['word'])
prob = prob.merge(prob_w_sachant_S_R, how='inner', left_on=['word', 'recipient'], right_on=['word', 'recipient'])
prob['log_prob_word_sachant_S_R'] = np.log(0.6*prob['prob_w_sachant_S_R']+ 0.2*prob['prob_w_sachant_R']+ 0.2*prob['prob_w'])

del Prob_R['mid']
print('Training Phase Done : call dataset "prob" to get prob for each word given recipient & sender ') 





