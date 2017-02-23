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
path_to_results= 'results/'

##########################
# load some of the files #
##########################

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)
test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

################################
# create some handy structures #
################################


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

model = Doc2Vec(alpha=0.025, min_alpha=0.025, size=100, window=5, min_count=5,
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
training_info['representation'] = training_info['body'].map(lambda x: model.infer_vector(x))
test_info['representation'] = test_info['body'].map(lambda x: model.infer_vector(x))

training_info.to_csv(path_to_results + 'training_info_embeddings.csv', sep=',', header=0)
test_info.to_csv(path_to_results + 'test_info_embeddings.csv', sep=',', header=0)
