# -*- coding: UTF-8 -*-

import os
import re
import torch
import logging
import pandas as pd
import pickle as pkl

from torch.autograd import backward
from Config import Config
from transformers import BertTokenizer, AlbertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from utils.OIJob import open_file, write_file
import torch.distributed as dist

class DatasetIterater(Dataset):
    """
    数据迭代器
    """
    def __init__(self, src, tgt, attention_mask):
        self.src = src
        self.tgt = tgt
        self.attention_mask = attention_mask

    def __getitem__(self, index):
        return self.src[index], self.attention_mask[index], self.tgt[index]

    def __len__(self):
        return len(self.src)


class DataManager(object):
    """
    数据处理类
    """
    def __init__(self):
        self.config = Config()
        
        if self.config.mode == 'train' and torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(backend='nccl', 
                                                 init_method=self.config.init_method, 
                                                #  init_method='env://',
                                                 rank=0, 
                                                 world_size=1)
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    
        self.sign_pad = '[PAD]'
        self.sign_unk = '[UNK]'
        self.sign_cls = '[CLS]'
        self.sign_sep = '[SEP]'
        # bert分词器
        # self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
        self.device = torch.device(self.config.device)
        self.path = self.config.path_dataset
        self.batch_size = self.config.batch_size
        self.path_tgt_map = self.config.path_tgt_map
        # 读取数据集
        dict_data = self.read_dataset(mode=self.config.mode)
        if self.config.mode == 'train':
            # 获取词表
            self.token2index, self.index2token = self.get_vocab_dictionary(dict_data)
            # 获取标签转换字典
            self.tgt2index, self.index2tgt = self.get_label_dictionary(dict_data)
        else:
            self.token2index, self.index2token = self.read_vocab_dictionary()
            self.tgt2index, self.index2tgt = self.read_label_dictionary()
        # 标签转换
        self.num_labels = len(self.tgt2index.keys())
        dict_data = self.transfer_label(dict_data, self.tgt2index)
        # 数据转换成tensor
        self.vocab2index = self.init_vocab(mode=self.config.mode)
        self.dict_data = self.data2tensor(dict_data)


    def init_vocab(self, mode):
        """初始化词表"""
        vocab = lambda x: self.token2index[x]            
        return vocab
    

    def word2index(self, x):
        """将词转换为index"""
        if self.token2index.get(x, -1) != -1:
            token = self.token2index[x]
        else:
            token = self.token2index[self.sign_unk]
        return token


    def read_dataset(self, mode='train'):
        """
        读取数据集
        """
        if mode == 'train':
            target = ['train.txt', 'dev.txt', 'test.txt']
        elif mode == 'eval':
            target = ['dev.txt']
        else:
            target = ['test.txt']
        # 遍历目标文件
        dict_data = {}
        for file in target:
            tmp_path = os.path.join(self.path, file)
            tmp_src, tmp_tgt = open_file(tmp_path, sep=' ')
            # 添加到数据字典
            key = file.split('.')[0]
            dict_data[key] = {'src':tmp_src, 'tgt':tmp_tgt}
        return dict_data


    def get_vocab_dictionary(self, dict_data):
        """获取训练集词表"""
        if self.config.model_name in self.config.model_list_nopretrain:
            src = dict_data['train']['src']
            words = [w for line in src for w in line if w != '']
            words = list(set(words))
            words = sorted(words, reverse=False)
            token2index = {x:i for i,x in enumerate(words)}
            index2token = {i:x for i,x in enumerate(words)}
            index2token[len(token2index)] = self.sign_pad
            token2index[self.sign_pad] = len(token2index)
            if self.sign_unk not in token2index.keys():
                index2token[len(token2index)] = self.sign_unk
                token2index[self.sign_unk] = len(token2index)
            if self.sign_cls not in token2index.keys():
                index2token[len(token2index)] = self.sign_cls
                token2index[self.sign_cls] = len(token2index)
            if self.sign_sep not in token2index.keys():
                index2token[len(token2index)] = self.sign_sep
                token2index[self.sign_sep] = len(token2index)
            # 标签映射表存到本地
            # write_file(token2index, self.config.path_vocab+'.txt')
            # pkl.dump(token2index, open(self.config.path_vocab, 'wb'))
        else:
            model_name = self.config.model_pretrain_online_checkpoint
            vocab = self.map_tokenizer(self.config.model_name).from_pretrained(model_name)
            if not os.path.exists(self.config.path_tokenizer):
                os.makedirs(self.config.path_tokenizer)
            vocab.save_vocabulary(self.config.path_tokenizer)
            vocabulary = vocab.get_vocab()
            token2index = vocabulary
            index2token = {i:x for x,i in vocabulary.items()}
            self.sign_unk = vocab.unk_token
            self.sign_pad = vocab.pad_token
        write_file(token2index, self.config.path_vocab+'.txt')
        pkl.dump(token2index, open(self.config.path_vocab, 'wb'))
        return token2index, index2token
    

    def read_vocab_dictionary(self):
        """读取训练集词表"""
        if self.config.model_name in self.config.model_list_nopretrain:
            tokenizer = pkl.load(open(self.config.path_vocab, 'rb'))
            index2token = {i:x for x,i in tokenizer.items()}
            token2index = {x:i for x,i in tokenizer.items()}
        else:
            model_name = self.config.model_pretrain_online_checkpoint
            vocab = self.map_tokenizer(self.config.model_name).from_pretrained(model_name)
            vocabulary = vocab.get_vocab()
            token2index = {x:i for i,x in enumerate(vocabulary)}
            index2token = {i:x for i,x in enumerate(vocabulary)}
            self.sign_unk = vocab.unk_token
            self.sign_pad = vocab.pad_token
        return token2index, index2token


    def get_label_dictionary(self, dict_data):
        """获取标签转换字典"""
        # 统计标签映射字典
        target_tgt = dict_data['train']['tgt']
        target_tgt = set([ x for line in target_tgt for x in line])
        tgt2index = {x:i for i, x in enumerate(target_tgt)}
        index2tgt = {i:x for i, x in enumerate(target_tgt)}
        # pad_index = tgt2index['O']
        # index2tgt[pad_index] = self.sign_pad
        # index2tgt[1] = self.sign_unk
        # tgt2index[self.sign_pad] = pad_index
        # 标签映射表存到本地
        write_file(tgt2index, self.config.path_tgt_map)
        return tgt2index, index2tgt


    def read_label_dictionary(self):
        """打开标签字典"""
        tgt2index = {}
        index2tgt = {}
        with open(self.config.path_tgt_map, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                tgt = line[0]
                index = int(line[1])
                tgt2index[tgt] = index
                index2tgt[index] = tgt
        return tgt2index, index2tgt


    def transfer_label(self, dict_data, tgt2index):
        """标签转换"""
        # 修改标签数据，标签转换成int
        for c, dt in dict_data.items():
            tmp_tgt = []
            for line in dt['tgt']:
                tmp_line = []
                for y in line:
                    tmp_line.append(tgt2index[y])
                tmp_tgt.append(tmp_line)
            dict_data[c]['tgt'] = tmp_tgt
        return dict_data


    def data2tensor(self, dict_data, pad=True):
        """
        将数据转换成tensor格式
        """
        dict_tensor = {}
        for dtype in dict_data.keys():
            src = dict_data[dtype]['src']
            tgt = dict_data[dtype]['tgt']
            assert len(src) == len(tgt), "length is not equation between src and tgt."
            # padding
            if pad:
                max_seq_length = self.config.max_seq_length - 2
                src = [ [x for x in line] for line in src]
                src = [x[:max_seq_length] if len(x) >= max_seq_length else x + [self.sign_pad]*(max_seq_length-len(x))  for x in src]
                tgt = [x[:max_seq_length] if len(x) >= max_seq_length else x + [self.tgt2index['O']]*(max_seq_length-len(x))  for x in tgt]
            # 加上[CLS]/[SEP]字符
            src = [ [self.sign_cls]+line+[self.sign_sep] for line in src]
            tgt = [ [self.tgt2index['O']]+line+[self.tgt2index['O']] for line in tgt]
            # Input转换
            src_ids = [[self.word2index(x) for x in line] for line in src]      # token转换成index
            attention_mask = [[1 if x != self.sign_pad else 0 for x in line] for line in src]           # attention matrix
            # 转换成tensor
            tensor_src_ids = torch.LongTensor(src_ids).to(self.device)
            tensor_attention_mask = torch.LongTensor(attention_mask).to(self.device)
            tensor_tgt = torch.LongTensor(tgt).to(self.device)
            # assert tensor_src_ids.size()[1]==256 and tensor_attention_mask.size()[1]==256 and tensor_tgt.size()[1]==256, 'input lenght no equal'
            dict_tensor[dtype] = {
                'src' : tensor_src_ids,
                'attention_mask' : tensor_attention_mask,
                'tgt' : tensor_tgt
            }
        return dict_tensor
                

    def get_data_train(self):
        """获取训练集数据"""
        src = self.dict_data['train']['src']
        tgt = self.dict_data['train']['tgt']
        attention_mask = self.dict_data['train']['attention_mask']
        # data = TensorDataset(src, tgt, attention_mask)
        data = DatasetIterater(src, tgt, attention_mask)
        sampler = RandomSampler(data) if not torch.cuda.device_count() > 1 else DistributedSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.config.batch_size)    # collate_fn=self.scale
        return dataloader
        
    
    def get_data_valid(self):
        """获取验证集数据"""
        src = self.dict_data['dev']['src']
        tgt = self.dict_data['dev']['tgt']
        attention_mask = self.dict_data['dev']['attention_mask']
        # data = TensorDataset(src, tgt, attention_mask)
        data = DatasetIterater(src, tgt, attention_mask)
        sampler = RandomSampler(data) if not torch.cuda.device_count() > 1 else DistributedSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.config.batch_size)
        return dataloader


    def get_data_test(self):
        """获取测试集数据"""
        src = self.dict_data['test']['src']
        tgt = self.dict_data['test']['tgt']
        attention_mask = self.dict_data['test']['attention_mask']
        # data = TensorDataset(src, tgt, attention_mask)
        data = DatasetIterater(src, tgt, attention_mask)
        sampler = RandomSampler(data) if not torch.cuda.device_count() > 1 else DistributedSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.config.batch_size)
        return dataloader
    

    def map_tokenizer(self, name):
        """模型映射"""
        if name == 'albert_crf':
            tokenizer = AlbertTokenizer
        else:
            tokenizer = BertTokenizer
        return tokenizer

