# -*- coding: UTF-8 -*-

import re
import os 
import pandas as pd
import torch
import logging
import pickle as pkl
import torch.nn as nn
from metrics.f1 import SeqEntityScore



def eval(loader, model, id2label, func_index2token, markup='bios'):
    
    metric = SeqEntityScore(id2label, markup=markup)
    

    # 遍历每个batch
    for bs in loader:
        with torch.no_grad():
            # 输入
            input_ids = bs[0]
            att_mask = bs[1]
            labels = bs[2]

            # 输出
            if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
                model = model.module
            model.eval()
            outputs = model(input_ids)
            outputs = outputs[1]
            outputs = model.crf.decode(outputs, att_mask)                       # (1, batch_size, seq_size)
            predicts = torch.squeeze(outputs,dim=0).cpu().numpy().tolist()      # (batch_size, seq_size)
        # predicts = torch.argmax(outputs,dim=2).cpu().numpy().tolist()#[0]
        tgt_size = labels.size()[0]
        
        assert tgt_size == len(predicts), "valid set: length difference between tgt and pred, in batch:%s" %str(i)
        labels = labels.tolist()

        # index 转 label
        labels = [[id2label[x] for x in line if id2label[x] != '[PAD]'] for line in labels]
        predicts = [[id2label[x] for x in pre][:len(lab)] for pre, lab in zip(predicts,labels)]
        # predicts = [[id2label[x] for x in line if id2label[x] != '[PAD]'] for line in predicts]

        # BIMES转BIO
        labels_tranf, predicts_tranf = BIMES2BIO(labels, predicts)

        # 更新
        metric.update(labels_tranf, predicts_tranf)
        
    # 获取指标结果
    eval_info, entity_info = metric.result()
    for k, v in entity_info.items():
        print('metrics:  lab:{0}, precision:{1}  recall:{2}  f1:{3}'.format(k, v['acc'], v['recall'], v['f1']))
    print('metrics: precision:{0}  recall:{1}  f1:{2}'.format(eval_info['acc'], eval_info['recall'], eval_info['f1']))
    return eval_info['f1']



def BIMES2BIO(labels, predicts):
    """标注BIMES转BIO"""
    labels_tranf = []
    for line in labels:
        tmp_line = []
        for x in line:
            if re.findall('', x):
                tmp_x = re.sub('M-|E-', 'I-', x)
            elif re.findall('S-', x):
                tmp_x = re.sub('S-','B-',x)
            else:
                tmp_x = x
            tmp_line.append(tmp_x)
        labels_tranf.append(tmp_line)
    predicts_tranf = []
    for line in predicts:
        tmp_line = []
        for x in line:
            if re.findall('M-|E-', x):
                tmp_x = re.sub('M-|E-', 'I-', x)
            elif re.findall('S-', x):
                tmp_x = re.sub('S-','I-',x)
            else:
                tmp_x = x
            tmp_line.append(tmp_x)
        predicts_tranf.append(tmp_line)
    return labels_tranf, predicts_tranf
