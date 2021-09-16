# -*- coding: UTF-8 -*-

import re
import os
from numpy.core.numeric import Inf 
import pandas as pd
# from thinc import optimizers
from tqdm.auto import tqdm
import torch
import logging
import pickle as pkl
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import f1_score
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AlbertTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig

from arg_config import Config
from model.bert_crf import BertCRF
from model.roberta_crf import RoBertaCRF
from model.lstm_crf import LSTM_CRF
from model.transformer_crf import TransformerCRF
from model.albert_crf import AlbertCRF
from config.BertConfig import BertConfig
from config.AlbertConfig import AlbertConfig
from utils.progressbar import ProgressBar
# from DataManager import DataManager

from process.eval import eval


Config = Config()
device = torch.device(Config.device)

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    if 'transformer' in name:
                        nn.init.uniform_(w, -0.1, 0.1)
                        # if len(w.size()) > 1:
                        #     nn.init.kaiming_uniform_(w)
                        # else:
                        #     nn.init.uniform_(w, -0.1, 0.1)
                    else:
                        nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(train_loader, valid_loader, num_labels, index2tgt):
    """
    训练过程
    """

    # 读取模型
    print('loading model...%s' %Config.model_name)
    if Config.model_name in Config.model_list_nopretrain:
        tokenizer = pkl.load(open(Config.path_vocab, 'rb'))
        index2token = {i:x for x,i in tokenizer.items()}
        func_index2token = lambda x: index2token[x]
        model = map_model(Config.model_name)(Config)
        # 参数加载
        if Config.continue_train:
            # 读取已存在的模型参数
            model.load_state_dict(torch.load(Config.path_model + 'pytorch_model.bin'))
        else:
            # 重新初始化模型参数
            init_network(model)
    else:
        tokenizer = map_tokenizer(Config.model_name).from_pretrained(Config.model_pretrain_online_checkpoint)
        # tokenizer = BertTokenizer.from_pretrained(Config.model_pretrain_online_checkpoint)
        tokenizer.save_pretrained(Config.path_tokenizer)
        func_index2token = tokenizer.convert_ids_to_tokens
        model = map_model(Config.model_name)
        # 参数加载
        if Config.continue_train:
            # 读取已存在的模型参数
            model = model.from_pretrained(Config.model_pretrain_online_checkpoint)
        else:
            # 加载预训练模型
            model_config = AutoConfig.from_pretrained(Config.model_pretrain_online_checkpoint, num_labels=num_labels)#num_labels=len(tgt2index.keys())-1)
            model = model.from_pretrained(Config.model_pretrain_online_checkpoint, config=model_config)
        
        # 冻结base model的参数
        if Config.model_pretrain_trainable:
            for param in model.base_model.parameters():
                param.requires_grad = True
                
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
    
    # 配置优化器
    t_total= Config.epoch * len(train_loader)
    warmup_steps = int(t_total * Config.warmup_proportion)

    # warm up & weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        # crf的参数优化方式
        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': Config.weight_decay, 'lr': Config.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': Config.crf_learning_rate},
        # 分类层的参数优化方式
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': Config.weight_decay, 'lr': Config.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': Config.crf_learning_rate}
    ]
    if Config.model_name in Config.model_list_pretrain:
        bert_param_optimizer = list(model.base_model.named_parameters())
        optimizer_grouped_parameters += [
        # bert的参数优化方式
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': Config.weight_decay, 'lr': Config.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': Config.learning_rate}
        ] 

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.learning_rate, eps=Config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    # optimizer = AdamW(model.parameters(), lr=Config.learning_rate)

    # 多卡训练
    model.to(device)
    if torch.cuda.device_count() > 1:
        # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
        model = nn.parallel.DistributedDataParallel(model, 
                                                    find_unused_parameters=True,
                                                    broadcast_buffers=True)
        # model = nn.DataParallel(model)
        
    print('start training..')
    best_f1 = 0
    global_step = 0
    # progress_bar = tqdm(range(len(train_loader)*Config.epoch))
    for epo in range(Config.epoch):
        progress_bar = ProgressBar(n_total=len(train_loader), desc='Training epoch:{0}'.format(epo))
        for i, bs in enumerate(train_loader):
            # print('current step:%s  warmup_steps:%s/%s  learning_rate:%s'%(global_step, warmup_steps,t_total,scheduler.optimizer.param_groups[0]['lr']))
            # 启动训练模式
            model.train()
            # 输入
            input_ids = bs[0]
            attention_mask = bs[1]
            labels = bs[2]

            # 定义loss，并训练
            # optimizer.zero_grad()   # 梯度清零
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)   #
            loss = outputs[0]           # 获取每个token的logit输出结果
            loss = loss.mean()
            # print(loss)
            
            loss.backward()             
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()   # 梯度清零
            global_step += 1       
            # progress_bar.update(1)
            progress_bar(i, {'loss': loss.item()})
            
            # 验证效果
            if global_step % Config.step_save==0 and global_step>0:
                f1 = eval(valid_loader, model, index2tgt, func_index2token)
                print('current epoch: %s/%s  iter:%s  loss:%.6f  valid f1:%.3f ' %(epo, Config.epoch, i, loss.item(), f1))
                # 模型保存
                if f1 > best_f1:
                    # steps_no_save = global_step
                    save_path = os.path.join(Config.path_model, 'epoch_'+str(epo))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    if Config.model_name in Config.model_list_nopretrain:
                        torch.save(model.state_dict(), Config.path_model+'pytorch_model.bin')
                    else:
                        model_save = model.module if torch.cuda.device_count() > 1 else model
                        model_save.save_pretrained(save_path)
                    print('save model success! ')
                    if not os.path.exists(Config.path_optimizer):
                        os.mkdir(Config.path_optimizer)
                    torch.save(optimizer.state_dict(), os.path.join(Config.path_optimizer, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(Config.path_optimizer, "scheduler.pt"))
                    print("Saving optimizer and scheduler states to %s"%Config.path_model)
        # 验证效果
        f1 = eval(valid_loader, model, index2tgt, func_index2token)
        print('current epoch: %s/%s  loss:%.6f  valid f1:%.3f ' %(epo, Config.epoch, loss.item(), f1))
    print('training end..')


def get_tag_index(labels, string_tag, tag_method='BMES'):
    """
    获取满足标注规则，标签的index
    labels: 包含标注类别的list
    string_tag: 标注的类别，如NAME, LOC等
    tag_method: 标注方法，BIO/BMES
    """
    b = 'B-'+string_tag
    m = 'M-'+string_tag
    e = 'E-'+string_tag
    s = 'S-'+string_tag
    o = 'O'

    target = []
    tmp = []
    for i, lab in enumerate(labels):
        if b == lab:
            tmp.append(i)
        elif len(tmp) and m == lab:
            tmp.append(i)
        elif len(tmp) and e == lab:
            tmp.append(i)
            tmp = [tmp[0]] + [tmp[-1]]
            target.append(tmp)
            tmp = []
        elif not len(tmp) and s == lab:
            tmp = [i]
            target.append(tmp)
            tmp = []
    return target


def map_model(name):
    """模型映射"""
    if name == 'lstm_crf':
        model = LSTM_CRF
    elif name == 'transformer':
        model = TransformerCRF
    elif name == 'bert_crf':
        model = BertCRF
    elif name == 'roberta_crf':
        model = RoBertaCRF
    elif name == 'albert_crf':
        model = AlbertCRF
    else:
        model = BertCRF
    return model


def map_tokenizer(name):
    """模型映射"""
    if name == 'albert_crf':
        tokenizer = AlbertTokenizer
    else:
        tokenizer = BertTokenizer
    return tokenizer
        

if __name__ == '__main__':
    train()

