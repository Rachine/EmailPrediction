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

# create address book with frequency information for each user
address_books = {}
i = 0

for sender, ids in emails_ids_per_sender.iteritems():
    recs_temp = []
    for my_id in ids:
        recipients = training_info[training_info['mid']==int(my_id)]['recipients'].tolist()
        recipients = recipients[0].split(' ')
        # keep only legitimate email addresses
        recipients = [rec for rec in recipients if '@' in rec]
        recs_temp.append(recipients)
    # flatten
    recs_temp = [elt for sublist in recs_temp for elt in sublist]
    # compute recipient counts
    rec_occ = dict(Counter(recs_temp))
    # order by frequency
    sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse = True)
    # save
    address_books[sender] = sorted_rec_occ

    if i % 10 == 0:
        print i
    i += 1

# save all unique recipient names
all_recs = list(set([elt[0] for sublist in address_books.values() for elt in sublist]))

# save all unique user names
all_users = []
all_users.extend(all_senders)
all_users.extend(all_recs)
all_users = list(set(all_users))

#############
# baselines #
#############

# will contain email ids, predictions for random baseline, and predictions for frequency baseline
predictions_per_sender = {}

# number of recipients to predict
k = 10

for index, row in test.iterrows():
    name_ids = row.tolist()
    sender = name_ids[0]
    # get IDs of the emails for which recipient prediction is needed
    ids_predict = name_ids[1].split(' ')
    ids_predict = [int(my_id) for my_id in ids_predict]
    random_preds = []
    freq_preds = []
    # select k most frequent recipients for the user
    k_most = [elt[0] for elt in address_books[sender][:k]]
    for id_predict in ids_predict:
        # select k users at random
        random_preds.append(random.sample(all_users, k))
        # for the frequency baseline, the predictions are always the same
        freq_preds.append(k_most)
    predictions_per_sender[sender] = [ids_predict,random_preds,freq_preds]

#################################################
# write predictions in proper format for Kaggle #
#################################################

path_to_results = 'results/'

with open(path_to_results + 'predictions_random.txt', 'wb') as my_file:
    my_file.write('mid,recipients' + '\n')
    for sender, preds in predictions_per_sender.iteritems():
        ids = preds[0]
        random_preds = preds[1]
        for index, my_preds in enumerate(random_preds):
            my_file.write(str(ids[index]) + ',' + ' '.join(my_preds) + '\n')

with open(path_to_results + 'predictions_frequency.txt', 'wb') as my_file:
    my_file.write('mid,recipients' + '\n')
    for sender, preds in predictions_per_sender.iteritems():
        ids = preds[0]
        freq_preds = preds[2]
        for index, my_preds in enumerate(freq_preds):
            my_file.write(str(ids[index]) + ',' + ' '.join(my_preds) + '\n')
