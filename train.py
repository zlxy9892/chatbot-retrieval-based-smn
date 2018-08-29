# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
from keras.preprocessing.sequence import pad_sequences
from utils import Utils
from retrival_model import SMN


### data preprocess
utils = Utils()

print('fetching all documents and word2id & id2word...')
# x_docs, y_docs = utils.get_x_y(f_qaqaq='./data/QAQAQ.txt', f_a='./data/A.txt')
# utils.pickle_save_data([x_docs, y_docs], './data/xy_docs.pkl')
(x_docs, y_docs) = utils.pickle_load_data(2, './data/xy_docs.pkl')
all_docs = list(x_docs + y_docs)
id2word, word2id = utils.extract_character_vocab(all_docs)
vocab_size = len(id2word)
print('vocab size: {}'.format(vocab_size))

print('fetching all sequences...')
# all_sequences = utils.get_all_sequences('./data/QAQAQ.txt', word2id)
# utils.pickle_save_data([all_sequences], './data/all_sequences.pkl')
(all_sequences,) = utils.pickle_load_data(1, './data/all_sequences.pkl')
print(np.shape(all_sequences))
sequences_len = [len(sequence) for sequences in all_sequences for sequence in sequences]
print('max_sequence_length: {}'.format(np.max(sequences_len)))
print('mean_sequence_length: {}'.format(np.mean(sequences_len)))
print('min_sequence_length: {}'.format(np.min(sequences_len)))
# utterances, utterances_length = utils.multi_sequences_padding(all_sequences, max_sentence_len=20)
# print(utterances[:2])
# print(utterances_length[:2])

print('fetching all responses...')
# all_responses_true = utils.get_all_responeses('./data/A.txt', word2id)
# utils.pickle_save_data([all_responses_true], './data/all_responses_true.pkl')
(all_responses_true,) = utils.pickle_load_data(1, './data/all_responses_true.pkl')

# print('removing short sequences...')
# all_sequences, all_responses_true = utils.remove_short_sequence(all_sequences, all_responses_true, 1)

print(len(all_sequences))
print(len(all_responses_true))
sequences_len = [len(sequence) for sequences in all_sequences for sequence in sequences]
print('max_sequence_length: {}'.format(np.max(sequences_len)))
print('mean_sequence_length: {}'.format(np.mean(sequences_len)))
print('min_sequence_length: {}'.format(np.min(sequences_len)))

print(utils.get_sentence_from_ids(all_responses_true[0], id2word, sep=' / '))
responses_len = utils.get_sequences_length(all_responses_true, maxlen=20)
print(responses_len[:10])
print(all_responses_true[:2])

all_responses_len = np.array(utils.get_sequences_length(all_responses_true, maxlen=20))
print('max_response_length: {}'.format(np.max(all_responses_len)))
print('mean_response_length: {}'.format(np.mean(all_responses_len)))
print('min_response_length: {}'.format(np.min(all_responses_len)))



### start training
print('\nstart training the model...\n')
smn = SMN(max_num_utterance=5,
          max_sentence_len=20,
          total_words=vocab_size)
smn.build_model()
smn.train_model(all_sequences, all_responses_true)



