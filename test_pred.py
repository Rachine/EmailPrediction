#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 21:17:36 2017

@author: paulinenicolas
"""
import pandas as pd
import itertools
import os

init_path = os.getcwd()
pathname_train_info = init_path + '/data/training_info.csv'
pathname_train_set = init_path + '/data/training_set.csv'

path_to_data = '/Users/paulinenicolas/Documents/M2_Data_Science/Advanced_text_and_Graphs/Project/data/'

test_set = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)
test_info = pd.read_csv(path_to_data+'test_info.csv',  sep=',', header=0)

#Deleting the email content
#df_info = df_info[['mid', 'recipients']]

#Seperating the different mail id/recipients id
sender_info_test = pd.concat([pd.Series(row['sender'], row['mids'].split(' '))              
                    for _, row in test_set.iterrows()]).reset_index()

#renaming columns
sender_info_test.columns = ['mid', 'sender']
#Merging two dataset : final granularity : sender mail - recipient mail - number of mail sent from sendr to recipient
send_recip_nb_mail_test = sender_info_test.groupby(['sender'], as_index=False).count()

mid_unique = pd.DataFrame(sender_info_test['mid'].unique())
mid_unique.columns = ['mid']



#Appendix

cartesian_mid_recipient = [[x, y] for x in sender_info_test['mid'].unique() for y in recipient_info['recipient'].unique()]
df_mid_recipient = pd.DataFrame(cartesian_mid_recipient)
df_mid_recipient.columns = ['mid', 'recipient']

############################

print('New pandas Dataset created called send_recip_nb_mail')


import nltk
from nltk.stem.porter import *
import string

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


data_test = []
i=0
for text in test_info['body'].tolist():   
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
    print(i)
    data_test.append(doc2)   # list data contains the preprocessed documents
    
df1_test = pd.DataFrame({'word split': data_test})
df_word_test = pd.concat([test_info, df1_test], axis=1, join='inner')
del df_word_test['body']


#Granularity one word by line
t = df_word_test['word split'].str.split(' ').apply(pd.Series, 1).stack()
t.index = t.index.droplevel(-1)
t.name = 'word'
del df_word_test['word split']
df_word3_test = df_word_test.join(t)
del df_word3_test['date']

df_word3_test['mid'] = df_word3_test['mid'].astype(str)
df_word4_test = df_word3_test.merge(sender_info_test,  how='inner', left_on=['mid'], right_on=['mid'])
df_word5_test = df_word4_test.merge(prob[['word', 'sender', 'recipient', 'log_prob_word_sachant_S_R']],  how='inner', left_on=['sender', 'word'], right_on=['sender','word'])
df_word6_test = df_word5_test[['mid', 'word', 'sender', 'recipient', 'log_prob_word_sachant_S_R']].groupby(['mid', 'sender', 'recipient'], as_index=False).sum()

df_word6_test['prob_email_sachant_S_R'] = np.exp(df_word6_test['log_prob_word_sachant_S_R'])

del df_word6_test['log_prob_word_sachant_S_R']
df_word7_test = df_word6_test.merge(Prob_S_sachant_R_freq, how = 'inner', left_on=['sender', 'recipient'], right_on=['sender', 'recipient'] )

df_word8_test = df_word7_test.merge(Prob_R, how = 'inner', left_on=['recipient'], right_on=['recipient'] )

df_word8_test['final_prob']= df_word8_test['prob_S_R']
df_word8_test = df_word8_test.merge(df_mid_recipient, how = 'outer', left_on = ['recipient', 'mid'], right_on = ['recipient', 'mid'])
df_word8_test = df_word8_test.fillna(0)


df_word9_test=df_word8_test.sort(columns=['mid', 'final_prob'], ascending=False)

df_word9_test = df_word9_test.groupby('mid').head(10)
df_word10_test = df_word9_test[['mid', 'recipient']]
df_word10_test = df_word10_test.groupby('mid')['recipient'].apply(' '.join)
df_word10_test.columns = ['recipients']
df_word10_test.to_csv(path_to_results+'/result.csv', sep=',')
