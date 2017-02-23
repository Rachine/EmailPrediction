# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:51:53 2017

@author: Pauline Nicolas Leo Treguer Rachid Riad 
"""

# Embedding method with Doc2vec on Gensim


from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

import random
import operator
import pandas as pd
from collections import Counter

path_to_data = 'data/'

##########################
# load some of the files #
##########################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

################################
# create some handy structures #
################################

# convert training set to dictionary
emails_ids_per_sender = {}
for index, series in training.iterrows():
    row = series.tolist()
    sender = row[0]
    ids = row[1:][0].split(' ')
    emails_ids_per_sender[sender] = ids

# save all unique sender names
all_senders = emails_ids_per_sender.keys()

i = 0

emails_content = training_info[['mid','body']]
mids,messages = [],[]
for el in emails_content.values:
    mids.append(el[0])
    messages.append(el[1])


class LabeledLineSentence(object):
    def __init__(self, df):
        self.df = df[['mid','body']]
    def __iter__(self):
        for el in self.df.values:
            yield LabeledSentence(words=el[1].split(), tags=['TXT_%s' % el[0]])


sentences = LabeledLineSentence(emails_content)

model = Doc2Vec(alpha=0.025, min_alpha=0.025, size=50, window=5, min_count=5,
                dm=1, workers=8, sample=1e-5)

model.build_vocab(sentences)

for epoch in range(10):
    try:
        print 'epoch %d' % (epoch)
        model.train(sentences)
        model.alpha *= 0.99
        model.min_alpha = model.alpha
    except (KeyboardInterrupt, SystemExit):
        break
    
model.init_sims(replace=True)
model.save_word2vec_format('results/text.model.bin', binary=True)