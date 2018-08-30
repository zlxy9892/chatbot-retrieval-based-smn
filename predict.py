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
max_num_utterance = 5
negative_samples = 1
max_sentence_len = 25
word_embedding_size = 100
rnn_units = 200


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
print('\nstart predicting...\n')
smn = SMN(device_name=device_name,
          max_num_utterance=max_num_utterance,
          max_sentence_len=max_sentence_len,
          word_embedding_size=word_embedding_size,
          rnn_units=rnn_units)
smn.build_model()

model_file = './model/model-3400'

actions = all_responses_true[:]
history, history_len = utils.multi_sequences_padding(all_sequences, 20)
true_utt_len = np.array(utils.get_sequences_length(all_responses_true, maxlen=20))
true_utt = np.array(pad_sequences(all_responses_true, padding='post', maxlen=20))
actions_len = np.array(utils.get_sequences_length(actions, maxlen=20))
actions = np.array(pad_sequences(actions, padding='post', maxlen=20))
history, history_len = np.array(history), np.array(history_len)

low = 0
n_sample = 30
negative_samples = 1
negative_indices = [np.random.randint(0, actions.shape[0], n_sample) for _ in range(negative_samples)]
negs = [actions[negative_indices[i], :] for i in range(negative_samples)]
negs_len = [actions_len[negative_indices[i]] for i in range(negative_samples)]

dev_utterances = np.concatenate([history[low:low + n_sample]] * (negative_samples + 1), axis=0)
dev_responses = np.concatenate([true_utt[low:low + n_sample]] + negs, axis=0)
dev_utterances_len = np.concatenate([history_len[low:low + n_sample]] * (negative_samples + 1), axis=0)
dev_responses_len = np.concatenate([true_utt_len[low:low + n_sample]] + negs_len, axis=0)
y_true = np.concatenate([np.ones(n_sample)] + [np.zeros(n_sample)] * negative_samples, axis=0)

y_pred_proba, y_pred = smn.predict(model_file, dev_utterances, dev_responses, dev_utterances_len, dev_responses_len)
print(y_pred_proba)
print(y_pred)
