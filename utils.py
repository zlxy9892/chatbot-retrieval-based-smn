#!/usr/bin/env python
# coding:utf-8

import codecs
from segment import Seg
from gensim import corpora
from collections import defaultdict
import re
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence


class Utils(object):

    def __init__(self):
        self.seg = Seg(file_stopwords='./data/stopword_small.txt')
    
    def pickle_save_data(self, data_list, f_save):
        print('Starting pickle to save file...')
        with open(f_save, 'wb') as f:
            for data in data_list:
                pickle.dump(data, f)
        print('Pickle save finished')
    
    def pickle_load_data(self, data_size, f_save):
        print('Starting pickle to load file...')
        data_list = []
        with open(f_save, 'rb') as f:
            for i in range(data_size):
                data = pickle.load(f)
                data_list.append(data)
        print('Pickle load finished')
        return tuple(data_list)
    
    def generate_char_x_y(self, data_size=10000, seed=None):
        np.random.seed(seed)
        x = []
        y = []
        char_list = np.array([chr(i) for i in range(97, 123)])
        for _ in range(data_size):
            word_len = np.random.randint(3, 10+1, 1)
            rand_char_idx_list = np.random.randint(0, len(char_list), word_len)
            word = char_list[rand_char_idx_list]
            sorted_word = sorted(word)
            x.append(list(word))
            y.append(list(sorted_word))
        return x, y
    
    def get_x_y(self, f_qaqaq='./data/QAQAQ.txt', f_a='./data/A.txt'):
        seg_x = Seg(file_stopwords='./data/stopword_small.txt')
        seg_y = Seg(file_stopwords='./data/stopword_small.txt')
        docs_x = self.get_sentence_list(f_qaqaq)
        docs_y = self.get_sentence_list(f_a)
        data_size = len(docs_x)
        # data_size = 1000
        x = []
        y = []
        for i in range(data_size):
            word_list_x = seg_x.cut(docs_x[i])
            word_list_y = seg_y.cut(docs_y[i])
            x.append(list(word_list_x))
            y.append(list(word_list_y))
        return x, y
    
    def get_x_dev(self, f_qaqaq='./data/test50/questions50.txt'):
        seg_x = Seg(file_stopwords='./data/stopword_small.txt')
        docs_x = self.get_sentence_list(f_qaqaq)
        data_size = len(docs_x)
        x = []
        for i in range(data_size):
            word_list_x = seg_x.cut(docs_x[i])
            x.append(list(word_list_x))
        return x

    def extract_character_vocab(self, data):
        data = list(data)
        special_words = ['<PAD>', '<GO>', '<EOS>', '<UNK>']
        word_set = sorted(set([word for word_list in data for word in word_list]))
        id2word = {idx:word for idx,word in enumerate(special_words + word_set)}
        word2id = {word:idx for idx,word in id2word.items()}
        return id2word, word2id
    
    def pad_sentence_batch(self, sentence_batch, pad_id):
        '''
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
        '''
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_id] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def batch_iter(self, num_epochs, xs, ys, batch_size, x_pad_int, y_pad_int, shuffle=True):
        xs = np.array(xs)
        ys = np.array(ys)
        data_size = len(xs)
        num_batches_per_epoch = int((data_size-1)/batch_size)+1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_xs = xs[shuffle_indices]
                shuffled_ys = ys[shuffle_indices]
            else:
                shuffled_xs = xs
                shuffled_ys = ys
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num*batch_size
                end_index = min((batch_num+1)*batch_size, data_size)
                x_batch = shuffled_xs[start_index:end_index]
                y_batch = shuffled_ys[start_index:end_index]
                pad_x_batch = np.array(self.pad_sentence_batch(x_batch, x_pad_int))
                pad_y_batch = np.array(self.pad_sentence_batch(y_batch, y_pad_int))
                x_lengths = []
                for one_x in x_batch:
                    x_lengths.append(len(one_x))
                y_lengths = []
                for one_y in y_batch:
                    y_lengths.append(len(one_y))
                yield pad_x_batch, pad_y_batch, x_lengths, y_lengths
    
    def get_feed_in_data(self, xs, ys, x_pad_int, y_pad_int):
        xs = np.array(xs)
        ys = np.array(ys)
        pad_x_batch = np.array(self.pad_sentence_batch(xs, x_pad_int))
        pad_y_batch = np.array(self.pad_sentence_batch(ys, y_pad_int))
        x_lengths = []
        for one_x in xs:
            x_lengths.append(len(one_x))
        y_lengths = []
        for one_y in ys:
            y_lengths.append(len(one_y))
        return pad_x_batch, pad_y_batch, x_lengths, y_lengths
    
    def get_sentence_from_ids(self, ids, id2word, sep='', remove_special_words=True, replace_sp_words=False):
        res_word_list = []
        special_words = ['<PAD>', '<GO>', '<EOS>', '<UNK>']
        ids = list(ids)
        word_list = [id2word[one_id] for one_id in ids]
        res_word_list = word_list
        if remove_special_words:
            res_word_list = []
            for word in word_list:
                if word not in special_words:
                    res_word_list.append(word)
        if replace_sp_words:
            sp_code_word_dict = self.get_sp_code_word_dict()
            res_word_list_replaced = []
            for word in res_word_list:
                if word in list(sp_code_word_dict.keys()):
                    res_word_list_replaced.append(sp_code_word_dict[word])
                else:
                    res_word_list_replaced.append(word)
                res_word_list = res_word_list_replaced
        sentence = sep.join(res_word_list)
        return sentence
    
    def get_sentence_list(self, f_docs='./data/QAQAQ.txt'):
        with open(f_docs, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            docs = [line.strip() for line in lines]
        return docs
    
    ### for SMN
    def get_all_sequences(self, f_docs='./data/QAQAQ.txt', word2id=None):
        docs = self.get_sentence_list(f_docs=f_docs)
        all_sequences = []
        for qaqaq_sentence in docs:
            utterances = self.get_qa_list(qaqaq_sentence)
            utterances_ids = [[word2id.get(word, word2id['<UNK>'])
                                for word in self.seg.cut(utterance)] for utterance in utterances]
            all_sequences.append(utterances_ids)
        return all_sequences
    
    def get_all_responeses(self, f_docs='./data/A.txt', word2id=None):
        docs = self.get_sentence_list(f_docs=f_docs)
        responses_ids = [[word2id.get(word, word2id['<UNK>'])
                        for word in self.seg.cut(doc)] for doc in docs]
        return responses_ids

    def multi_sequences_padding(self, all_sequences, max_sentence_len=50):
        max_num_utterance = 5
        PAD_SEQUENCE = [0] * max_sentence_len
        padded_sequences = []
        sequences_length = []
        for sequences in all_sequences:
            sequences_len = len(sequences)
            sequences_length.append(self.get_sequences_length(sequences, maxlen=max_sentence_len))
            if sequences_len < max_num_utterance:
                sequences += [PAD_SEQUENCE] * (max_num_utterance - sequences_len)
                sequences_length[-1] += [0] * (max_num_utterance - sequences_len)
            else:
                sequences = sequences[-max_num_utterance:]
                sequences_length[-1] = sequences_length[-1][-max_num_utterance:]
            sequences = pad_sequences(sequences, padding='post', maxlen=max_sentence_len)
            padded_sequences.append(sequences)
        return padded_sequences, sequences_length
    
    def get_sequences_length(self, sequences, maxlen):
        sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
        return sequences_length
    
    def remove_short_sequence(self, all_sequences, all_responeses, sequence_len_min=9):
        if len(all_sequences) != len(all_responeses):
            print('length of all_sequences and all_responses does not equal.')
            return None
        good_idx = []
        for i in range(len(all_sequences)):
            flag = True
            for sequence in all_sequences[i]:
                if len(sequence) < sequence_len_min:
                    flag = False
                    break
            if flag:
                response = all_responeses[i]
                if len(response) < sequence_len_min:
                    flag = False
            if flag:
                good_idx.append(i)
        res_all_sequences = [value for idx, value in enumerate(all_sequences) if idx in good_idx]
        res_all_responeses = [value for idx, value in enumerate(all_responeses) if idx in good_idx]
        return res_all_sequences, res_all_responeses
        
    ### SMN end
    
    def show_best_answers(self, f_answers='./data/test50/answers50.txt', answers_count=50):
        with codecs.open(f_answers, 'r', 'utf-8') as rf_standardAnswer:
            #10个标准答案
            standardAnswerLines = rf_standardAnswer.readlines()
            for i in range(answers_count):
                best_answer = ''
                highest_score = 0
                for j in range(10):
                    standardAnswerLine = standardAnswerLines[i*11+j+1].strip().split('\t')
                    reference_answer = standardAnswerLine[0].strip()
                    standardScore = float(standardAnswerLine[1])
                    if highest_score < standardScore:
                        highest_score = standardScore
                        best_answer = reference_answer
                print('\n------------------------------\n')
                print('{}:'.format(i+1))
                print(best_answer)
                print(highest_score)
    
    def get_best_answers_docs(self, f_answers='./data/test50/answers50.txt', answers_count=50):
        seg_y = Seg(file_stopwords='./data/stopword_small.txt')
        best_answers = []
        best_answers_docs = []
        with codecs.open(f_answers, 'r', 'utf-8') as rf_standardAnswer:
            standardAnswerLines = rf_standardAnswer.readlines()
            for i in range(answers_count):
                best_answer = ''
                highest_score = 0
                for j in range(10):
                    standardAnswerLine = standardAnswerLines[i*11+j+1].strip().split('\t')
                    reference_answer = standardAnswerLine[0].strip()
                    standardScore = float(standardAnswerLine[1])
                    if highest_score < standardScore:
                        highest_score = standardScore
                        best_answer = reference_answer
                best_answers.append(best_answer)
            for i in range(len(best_answers)):
                word_list = seg_y.cut(best_answers[i])
                best_answers_docs.append(list(word_list))
            return best_answers_docs
    
    def show_answers(self, f_answers):
        with codecs.open(f_answers, 'r', 'utf-8') as rf_standardAnswer:
            #10个标准答案
            answers = rf_standardAnswer.readlines()
            answers_count = len(answers)
            for i in range(answers_count):
                print('\n------------------------------\n')
                print('{}:'.format(i+1))
                print(answers[i].strip())
    
    def show_cuted_sentence(self, sentence):
        print('\noriginal sentence:\n')
        print(sentence)
        words = self.seg.cut(sentence)
        print('\ncuted sentence:\n')
        print(" / ".join(words))
        print('\n')

    def save_word_count(self, f_source='./data/QAQAQ.txt', f_save_dic='./data/wordcount.dict'):
        with open(f_source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        docs = [line.strip() for line in lines]
        word_list = []
        for sentence in docs:
            word_list_one_row = self.seg.cut_for_search(sentence)
            word_list.extend(word_list_one_row)
        word_count = defaultdict(int)
        for word in word_list:
            word_count[word] += 1
        sorted_word_count = sorted(word_count.items(), key=lambda d: d[1]) 
        f = open(f_save_dic, 'w')
        f.write(str(sorted_word_count))
        f.close()
        
    def load_dict(self, f_dict):
        f = open(f_dict, 'r')
        a = f.read()
        saved_dict = eval(a)
        f.close()
        saved_dict = dict(saved_dict)
        return saved_dict
    
    def show_dict(self, input_dict):
        for key in input_dict:
            print(str(key) + ': ' + str(input_dict[key]))
    
    def get_contained_keywords(self, keywords, sentence):
        contained_keywords = []
        words = self.seg.cut_for_search(sentence)
        for word in words:
            if word in keywords and word not in contained_keywords:
                contained_keywords.append(word)
        return contained_keywords

    def get_contained_keywords_sentences(self, keywords, f_docs='./data/QAQAQ.txt'):
        res_sentences = []
        idx_list = []
        re_keywords = '.*|.*'.join(keywords)
        re_keywords = '.*' + re_keywords + '.*'
        # re_keywords2 = ['.*' + keyword + '.*' for keyword in keywords]
        # print(re_keywords)
        docs = self.get_sentence_list(f_docs)
        for i in range(len(docs)):
            if re.match(re.compile(re_keywords), docs[i]):
                contained_sentence = docs[i]
                res_sentences.append(contained_sentence)
                idx_list.append(i)
            # flag = True
            # for re_keyword in re_keywords2:
            #     if not re.match(re.compile(re_keyword), docs[i]):
            #         flag = False
            #         break
            # if flag:
            #     res_sentences.append(docs[i])
            #     idx_list.append(i)
                # print(docs[i])
        return res_sentences, idx_list
    
    def get_qa_list(self, qaqaq_sentence):
        qaqaq_sentence = qaqaq_sentence.strip()
        qaqaq = qaqaq_sentence.split('<s>')[:-1]
        return qaqaq
    
    def get_qq_aa(self, qaqaq_sentence):
        qaqaq_list = self.get_qa_list(qaqaq_sentence)
        qq = qaqaq_list[0] + '<s>' + qaqaq_list[2]
        aa = qaqaq_list[1] + '<s>' + qaqaq_list[3]
        return qq, aa

    def save_last_aq(self, f_qaqa='./data/test50/questions50.txt', f_save='./data/test50/questions50_aq.txt'):
        with open(f_qaqa, 'r', encoding='utf-8') as f_qaqaq:
            with open(f_save, 'w', encoding='utf-8') as f_save:
                qaqaq_list = f_qaqaq.readlines()
                for qaqaq_sentence in qaqaq_list:
                    qaqaq_sentence = qaqaq_sentence.strip()
                    qaqaq = qaqaq_sentence.split('<s>')[:-1]
                    aq = qaqaq[-2:]
                    f_save.write("{}<s>{}\n".format(aq[0], aq[1]))
    
    def write_sentences(self, sentences, f_save):
        with open(f_save, 'w', encoding='utf-8') as f_save:
            for sentence in sentences:
                sentence = sentence.strip()
                f_save.write("{}\n".format(sentence))
    
    def get_sp_code_word_dict(self):
        code_word_dict = {
            'COMMON1': '亲爱的请问还有其他可以帮到您?',
            'COMMON2': '请问还有其他还可以帮到您的吗?',
            'COMMON3': '尊敬的商家您好，我是您的京东物流小红人工号[数字x]',
            'COMMON4': '亲爱的客户辛苦您稍等一下下',
            'COMMON5': '可在手机端打开https?://[链接x]或在电脑端打开http://myivc.jd.com/fpzz.html进行发票查询和下载',
            'COMMON6': '请问您是要咨询订单:[ORDERID_?]',
            'COMMON7': '正在为您核实处理',
            'COMMON8': '申请路径:[站点x]可通过“我的京东”-“客户服务”-“返修退换货”内申请(也可直接点击此链接:http://myjd.jd.com/repair/orderlist.action;【APP端】可通过“我的”-“客户服务”-“退换/售后“中申请~',
            'LINK1': 'http://item.jd.com/[数字x].html',
            'LINK2': 'http://vc.jd.com/sampling.html',
            'LINK3': 'http://m-eve.jd.com/dxtyk/index体验卡',
            'LINK4': 'http://[链接x]',
            'LINK5': 'http://myjd.jd.com/repair/orderlist.action',
            'LINK6': 'http://myivc.jd.com/fpzz.html',
            'LINK7': 'http://rec.ql.jd.com/price/soplbpprice',
            'SPWORD1': '[订单编号:[ORDERID_?]，订单金额:[金额x]，下单时间:[日期x][时间x]]',
            'SPWORD2': '顾客通过点击web咚咚[站点x]信息发送:',
            'SPWORD4': '#E-s[数字x]',
            'SPWORD5': '¥[金额x]',
            'SPWORD6': '[日期x]', 
            'SPWORD7': '[时间x]', 
            'SPWORD8': '[姓名x]', 
            'SPWORD9': '[站点x]', 
            'SPWORD10': '[地址x]', 
            'SPWORD11': '[电话x]', 
            'SPWORD12': '[邮箱x]', 
            'SPWORD13': '[组织机构x]',
            'ORDER1': '[ORDERID_?]'  
        }
        return code_word_dict
