# -*- coding: UTF-8 -*-

import re
import os
from numpy.core.numeric import Inf 
import pandas as pd
from thinc import optimizers
import torch
import logging
import pickle as pkl
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import f1_score
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import AutoModel, AutoTokenizer, AutoConfig

from arg_config import Config
from model.bert_crf import BertCRF
from model.roberta_crf import RoBertaCRF
from model.lstm_crf import LSTM_CRF
from config.BertConfig import BertConfig
# from DataManager import DataManager
from metrics.WeightMacroF1 import WMF1
from metrics.f1 import SeqEntityScore

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
    if Config.model_name in ['lstm_crf']:
        tokenizer = pkl.load(open(Config.path_vocab, 'rb'))
        index2token = {i:x for x,i in tokenizer.items()}
        func_index2token = lambda x: index2token[x]
        model = LSTM_CRF(Config)
        init_network(model)
    else:
        tokenizer = BertTokenizer.from_pretrained(Config.model_name)
        tokenizer.save_pretrained(Config.path_tokenizer)
        func_index2token = tokenizer.convert_ids_to_tokens
        model_config = BertConfig.from_pretrained(Config.model_name, num_labels=num_labels)#num_labels=len(tgt2index.keys())-1)
        model = BertCRF.from_pretrained(Config.model_name, config=model_config)

        # 冻结base model的参数
        for param in model.base_model.parameters():
            param.requires_grad = False
    # 启动训练模式
    model.train()
    # 配置优化器
    t_total= Config.epoch * len(train_loader)
    warmup_steps = int(t_total * Config.warmup_proportion)


    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        # bert的参数优化方式
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': Config.weight_decay, 'lr': Config.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': Config.learning_rate},
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.learning_rate, eps=Config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    # optimizer = AdamW(model.parameters(), lr=Config.learning_rate)

    # 切换到gpu
    model.to(device)

    print('start training..')
    best_f1 = 0
    best_loss = 10000
    global_step = 0
    steps_trained_in_current_epoch = 0
    for epo in range(Config.epoch):
        i = 0
        for bs in train_loader:
            # print('current step:%s  warmup_steps:%s/%s  learning_rate:%s'%(global_step, warmup_steps,t_total,scheduler.optimizer.param_groups[0]['lr']))

            # 输入
            input_ids = bs[0]
            attention_mask = bs[1]
            labels = bs[2]

            # 定义loss，并训练
            # optimizer.zero_grad()   # 梯度清零
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)   #
            loss = outputs[0]           # 获取每个token的logit输出结果
            
            scheduler.step()  # Update learning rate schedule
            loss.backward()             
            optimizer.step()
            optimizer.zero_grad()   # 梯度清零
            global_step += 1       
            
            # 验证效果
            if i % 100 == 0:
                # f1 = get_score(valid_loader, model, index2tgt, tokenizer)
                f1 = eval(valid_loader, model, index2tgt, func_index2token)
                print('current epoch: %s/%s  iter:%s/%s  loss:%.6f  valid f1:%.3f ' %(epo, Config.epoch, i, bs[0].size()[0], loss.item(), f1))
            if global_step % 1000 == 0:
                # 模型保存
                if loss < best_loss:
                    # save_path = os.path.join(Config.path_save_model, 'epoch_'+str(epo))
                    model.save_pretrained(Config.path_model)
                    print('save model success! ')
                    torch.save(optimizer.state_dict(), os.path.join(Config.path_optimizer, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(Config.path_optimizer, "scheduler.pt"))
                    print("Saving optimizer and scheduler states to %s"%Config.path_model)

            # 测试效果
            if i % 500 == 0:
                # f1 = get_score(test_loader, model, index2tgt, tokenizer)
                f1 = eval(valid_loader, model, index2tgt, func_index2token)
                print('current epoch: %s/%s  iter:%s/%s  loss:%.6f  test f1:%.3f' %(epo, Config.epoch, i, bs[0].size()[0], loss.item(), f1))
            i += 1
        
        # 若本次迭代没有保存模型文件
        # save_path = os.path.join(Config.path_save_model, 'epoch_'+str(epo))
        # if not os.path.exists(save_path):
        #     model.save_pretrained(save_path)
        
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



if __name__ == '__main__':
    train()

