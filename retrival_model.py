# coding:utf-8

import time
import datetime
import os
import tensorflow as tf
import pickle
import utils
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import evaluate
from utils import Utils


class SMN():
    def __init__(self,
                 lr=0.001,
                 max_num_utterance=5,
                 negative_samples=1,
                 max_sentence_len=20,
                 word_embedding_size=100,
                 rnn_units=100,
                 total_words=66958,
                 batch_size=32,
                 num_checkpoints=10,
                 evaluate_every=100,
                 checkpoint_every=100):
        self.utils = Utils()
        self.lr = lr
        self.max_num_utterance = max_num_utterance
        self.negative_samples = negative_samples
        self.max_sentence_len = max_sentence_len
        self.word_embedding_size = word_embedding_size
        self.rnn_units = rnn_units
        self.total_words = total_words
        self.batch_size = batch_size
        self.num_checkpoints = num_checkpoints
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every

    def LoadModel(self):
        #init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        #with tf.Session() as sess:
            #sess.run(init)
        saver.restore(sess,"neg5model\\model.5")
        return sess
        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        # with tf.Session() as sess:
        #     # Restore variables from disk.
        #     saver.restore(sess, "/model/model.5")
        #     print("Model restored.")

    def build_model(self):
        # placeholders
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))
        self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
        self.y_true = tf.placeholder(tf.int32, shape=(None,))
        # self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
        self.response_len = tf.placeholder(tf.int32, shape=(None,))
        self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))

        # word_embedding vector
        word_embeddings = tf.get_variable('word_embeddings_v', initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1), shape=(self.total_words, self.word_embedding_size), dtype=tf.float32, trainable=True)
        # word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.word_embedding_size), dtype=tf.float32, trainable=False)
        # self.embedding_init = word_embeddings.assign(self.embedding_ph)

        # utterance embedding
        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1)
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)

        # response embedding
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)
        
        # GRU initialize
        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())

        # matrix 1
        A_matrix = tf.get_variable('A_matrix_v', shape=(self.rnn_units, self.rnn_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        reuse = None

        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, response_embeddings, sequence_length=self.response_len, dtype=tf.float32,
                                                       scope='sentence_GRU')
        self.response_embedding_save = response_GRU_embeddings
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])

        # generate matching vectors
        matching_vectors = []
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings)
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, utterance_embeddings, sequence_length=utterance_len, dtype=tf.float32,
                                                            scope='sentence_GRU')
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            conv_layer = tf.layers.conv2d(matrix, filters=8, kernel_size=(3, 3), padding='VALID',
                                          kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                          activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
                                                    padding='VALID', name='max_pooling')  # TODO: check other params
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh, reuse=reuse, name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)
        
        # last hidden layer
        _, last_hidden = tf.nn.dynamic_rnn(final_GRU, tf.stack(matching_vectors, axis=0, name='matching_stack'), dtype=tf.float32,
                                           time_major=True, scope='final_GRU')  # TODO: check time_major
        
        # output layer
        logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
        self.y_pred = tf.nn.softmax(logits)

        # loss
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits), name='loss')
        tf.summary.scalar('loss', self.loss)

        # optimize
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step, name='train_op')
    

    def Evaluate(self, sess):
        pass
        '''
        with open(evaluate_file, 'rb') as f:
           history, true_utt, labels = pickle.load(f)
        self.all_candidate_scores = []
        history, history_len = utils.multi_sequences_padding(history, self.max_sentence_len)
        history, history_len = np.array(history), np.array(history_len)
        true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=self.max_sentence_len))
        true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))
        low = 0
        while True:
            feed_dict = {
                self.utterance_ph: np.concatenate([history[low:low + 200]], axis=0),
                self.all_utterance_len_ph: np.concatenate([history_len[low:low + 200]], axis=0),
                self.response_ph: np.concatenate([true_utt[low:low + 200]], axis=0),
                self.response_len: np.concatenate([true_utt_len[low:low + 200]], axis=0),
            }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + 200
            if low >= history.shape[0]:
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)
        evaluate.ComputeR10_1(all_candidate_scores,labels)
        evaluate.ComputeR2_1(all_candidate_scores,labels)
        '''
    
    def train_model(self, all_sequences, all_responses_true, countinue_train=False, previous_modelpath="model"):
        # init = tf.global_variables_initializer()
        # saver = tf.train.Saver()
        # merged = tf.summary.merge_all()
        with tf.Session() as sess:
            # output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.curdir, 'log', timestamp))
            print('Writing log to {}\n'.format(out_dir))

            # summary all the trainable variables
            for var in tf.trainable_variables():
                tf.summary.histogram(name=var.name, values=var)

            # summaries for loss
            loss_summary = tf.summary.scalar('loss', self.loss)

            # train summaries
            train_summary_op = tf.summary.merge_all()
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, tf.get_default_graph())

            # dev summaries
            dev_summary_op = tf.summary.merge_all()
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, tf.get_default_graph())

            # checkpointing, tensorflow assumes this directory already existed, so we need to create it
            checkpoint_dir = os.path.join(out_dir, 'checkpoints')
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

            # initialize all variables
            sess.run(tf.global_variables_initializer())
            
            # get input data
            actions = all_responses_true[:]

            history, history_len = self.utils.multi_sequences_padding(all_sequences, self.max_sentence_len)
            true_utt_len = np.array(self.utils.get_sequences_length(all_responses_true, maxlen=self.max_sentence_len))
            true_utt = np.array(pad_sequences(all_responses_true, padding='post', maxlen=self.max_sentence_len))
            actions_len = np.array(self.utils.get_sequences_length(actions, maxlen=self.max_sentence_len))
            actions = np.array(pad_sequences(actions, padding='post', maxlen=self.max_sentence_len))
            history, history_len = np.array(history), np.array(history_len)
            
            low = 0
            epoch = 1
            while epoch < 10:
                n_sample = min(low + self.batch_size, history.shape[0]) - low
                negative_indices = [np.random.randint(0, actions.shape[0], n_sample) for _ in range(self.negative_samples)]
                negs = [actions[negative_indices[i], :] for i in range(self.negative_samples)]
                negs_len = [actions_len[negative_indices[i]] for i in range(self.negative_samples)]
                feed_dict = {
                    self.utterance_ph: np.concatenate([history[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                    self.all_utterance_len_ph: np.concatenate([history_len[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                    self.response_ph: np.concatenate([true_utt[low:low + n_sample]] + negs, axis=0),
                    self.response_len: np.concatenate([true_utt_len[low:low + n_sample]] + negs_len, axis=0),
                    self.y_true: np.concatenate([np.ones(n_sample)] + [np.zeros(n_sample)] * self.negative_samples, axis=0)
                }
                _, step, summaries, loss = sess.run(
                        [self.train_op, self.global_step, train_summary_op, self.loss],
                        feed_dict)
                timestr = datetime.datetime.now().isoformat()
                print('{}: => epoch {} | step {}| loss {:g}'.format(timestr, epoch, step, loss))
                train_summary_writer.add_summary(summaries, step)
                
                current_step = tf.train.global_step(sess, self.global_step)
                low += n_sample
                if current_step % self.evaluate_every == 0:
                    pass
                    # print("loss", sess.run(self.loss, feed_dict=feed_dict))
                    # self.Evaluate(sess)
                if current_step % self.checkpoint_every == 0:
                    saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)
                    path = saver.save(sess=sess, save_path=checkpoint_prefix, global_step=self.global_step)
                    print('\nSaved model checkpoint to {}\n'.format(path))
                if low >= history.shape[0]:
                    low = 0
                    epoch += 1


if __name__ == "__main__":
    smn = SMN()
    smn.build_model()
    # smn.train_model()
    #sess = scn.LoadModel()
    #scn.Evaluate(sess)
    #results = scn.BuildIndex(sess)
    #print(len(results))

    #scn.TrainModel()
