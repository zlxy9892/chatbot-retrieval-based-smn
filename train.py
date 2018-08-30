# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
from keras.preprocessing.sequence import pad_sequences
from utils import Utils
from retrieval_model import SMN

### hyper parameters
device_name = '/cpu:0'
lr = 0.001
max_num_utterance = 5
negative_samples = 1
max_sentence_len = 25
word_embedding_size = 100
rnn_units = 200
batch_size = 32
max_epoch = 100
num_checkpoints = 10
evaluate_every = 100
checkpoint_every = 100


### data preprocess
utils = Utils()

print('fetching all documents and word2id & id2word...')
x_docs, y_docs = utils.get_x_y(f_qaqaq='./data/QAQAQ.txt', f_a='./data/A.txt')
utils.pickle_save_data([x_docs, y_docs], './data/xy_docs.pkl')
# (x_docs, y_docs) = utils.pickle_load_data(2, './data/xy_docs.pkl')
all_docs = list(x_docs + y_docs)
id2word, word2id = utils.extract_character_vocab(all_docs)
vocab_size = len(id2word)
print('vocab size: {}'.format(vocab_size))

print('fetching all sequences...')
all_sequences = utils.get_all_sequences('./data/QAQAQ.txt', word2id)
utils.pickle_save_data([all_sequences], './data/all_sequences.pkl')
# (all_sequences,) = utils.pickle_load_data(1, './data/all_sequences.pkl')
print(np.shape(all_sequences))
sequences_len = [len(sequence) for sequences in all_sequences for sequence in sequences]
print('max_sequence_length: {}'.format(np.max(sequences_len)))
print('mean_sequence_length: {}'.format(np.mean(sequences_len)))
print('min_sequence_length: {}'.format(np.min(sequences_len)))
# utterances, utterances_length = utils.multi_sequences_padding(all_sequences, max_sentence_len=20)
# print(utterances[:2])
# print(utterances_length[:2])

print('fetching all responses...')
all_responses_true = utils.get_all_responeses('./data/A.txt', word2id)
utils.pickle_save_data([all_responses_true], './data/all_responses_true.pkl')
# (all_responses_true,) = utils.pickle_load_data(1, './data/all_responses_true.pkl')

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
smn = SMN(device_name=device_name,
          lr=lr,
          max_num_utterance=max_num_utterance,
          negative_samples=negative_samples,
          max_sentence_len=max_sentence_len,
          word_embedding_size=word_embedding_size,
          rnn_units=rnn_units,
          total_words=vocab_size,
          batch_size=batch_size,
          max_epoch=max_epoch,
          num_checkpoints=num_checkpoints,
          evaluate_every=evaluate_every,
          checkpoint_every=checkpoint_every)
smn.build_model()
# smn.train_model(all_sequences, all_responses_true, use_pre_trained=False)
# smn.train_model(all_sequences, all_responses_true, use_pre_trained=True, pre_trained_modelpath='./model/model-44800')

print('\n--- DONE! ---')
