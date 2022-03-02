
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
from apex import amp
from torch.nn import functional as F
from sklearn.metrics import f1_score
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AlbertTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.tensorboard import SummaryWriter

from Config import Config
from model.bert_crf import BertCRF
from model.roberta_crf import RoBertaCRF
from model.lstm_crf import LSTM_CRF
from model.transformer_crf import TransformerCRF
from model.albert_crf import AlbertCRF
from model.optimal.adversarial import FGM,PGD
from config.BertConfig import BertConfig
from config.AlbertConfig import AlbertConfig
from utils.progressbar import ProgressBar
# from DataManager import DataManager

from process.eval import eval
from process.MapModel import map_model, map_tokenizer



class Trainer(object):
    
    def __init__(self, 
                 config, 
                 train_loader, 
                 valid_loader, 
                 test_loader):
        self.config = config
        self.device = torch.device(self.config.device)
        # 加载数据集
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        # 加载标签
        self.tgt2index = { x.strip().split(' ')[0]:int(x.strip().split(' ')[1]) for x in open(self.config.path_tgt_map,'r').readlines()}
        self.index2tgt = { i:x for x,i in self.tgt2index.items()}
        self.num_labels = len(self.tgt2index.keys())
        # 加载模型
        self.load_model()


    def load_model(self):
        """
        加载模型
        """
        # print('model loading')
        # model_config = BertConfig.from_pretrained(self.config.initial_pretrain_model, num_labels=len(self.tag))
        # self.model = BertSpanForNer.from_pretrained(self.config.initial_pretrain_model, config=model_config)
        # self.model.to(self.device)
        # print('>>>>>>>> mdoel structure >>>>>>>>')
        # for name,parameters in self.model.named_parameters():
        #     print(name,':',parameters.size())
        # print('>>>>>>>> mdoel structure >>>>>>>>')
        
        # 读取模型
        print('loading model...%s' %self.config.model_name)
        if self.config.model_name in self.config.model_list_nopretrain:
            self.tokenizer = pkl.load(open(self.config.path_vocab, 'rb'))
            self.index2token = {i:x for x,i in self.tokenizer.items()}
            self.func_index2token = lambda x: self.index2token[x]
            self.model = map_model(self.config.model_name)(self.config)
            # 参数加载
            if self.config.continue_train:
                # 读取已存在的模型参数
                self.model.load_state_dict(torch.load(self.config.path_model + 'pytorch_model.bin'))
            else:
                # 重新初始化模型参数
                # if self.config.model_name in self.config.model_list_nopretrain:
                self.init_network(self.model)
        else:
            self.tokenizer = map_tokenizer(self.config.model_name).from_pretrained(self.config.model_pretrain_online_checkpoint)
            # tokenizer = BertTokenizer.from_pretrained(self.config.model_pretrain_online_checkpoint)
            self.tokenizer.save_pretrained(self.config.path_tokenizer)
            self.func_index2token = self.tokenizer.convert_ids_to_tokens
            self.model = map_model(self.config.model_name)
            # 参数加载
            if self.config.continue_train:
                # 读取已存在的模型参数
                self.model = self.model.from_pretrained(self.config.model_pretrain_online_checkpoint)
            else:
                # 加载预训练模型
                model_config = AutoConfig.from_pretrained(self.config.model_pretrain_online_checkpoint, num_labels=self.num_labels)#num_labels=len(tgt2index.keys())-1)
                self.model = self.model.from_pretrained(self.config.model_pretrain_online_checkpoint, config=model_config)    
            # # 冻结base model的参数
            # if self.config.model_pretrain_trainable:
            #     for param in model.base_model.parameters():
            #         param.requires_grad = True
        # 将模型加载到CPU/GPU
        self.model.to(self.device)

    
    def init_network(self, model, method='xavier', exclude='embedding', seed=123):
        """
        # 权重初始化，默认xavier
        """
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


    def train(self):
        """
        训练模型
        """
        # 打印模型参数
        for name,parameters in self.model.named_parameters():
            print(name,':',parameters.size())

        # warm up & weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        crf_param_optimizer = list(self.model.crf.named_parameters())
        linear_param_optimizer = list(self.model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            # crf的参数优化方式
            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay, 'lr': self.config.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.config.crf_learning_rate},
            # 分类层的参数优化方式
            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay, 'lr': self.config.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.config.crf_learning_rate}
        ]
        if self.config.model_name in self.config.model_list_pretrain:
            bert_param_optimizer = list(self.model.base_model.named_parameters())
            optimizer_grouped_parameters += [
            # bert的参数优化方式
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay, 'lr': self.config.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.config.learning_rate}
            ] 

        # 配置优化器
        t_total= self.config.epoch * len(self.train_loader)
        warmup_steps = int(t_total * self.config.warmup_proportion)
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        # optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)

        # 混合精度训练
        if self.config.fp16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.config.fp16_opt_level)

        # 多卡训练
        if torch.cuda.device_count() > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, 
                                                        find_unused_parameters=True,
                                                        broadcast_buffers=True)

        # 对抗训练
        if self.config.adv_option == 'FGM':
            self.fgm = FGM(self.model, emb_name=self.config.adv_name, epsilon=self.config.adv_epsilon)
        if self.config.adv_option == 'PGD':
            self.pgd = PGD(self.model, emb_name=self.config.adv_name, epsilon=self.config.adv_epsilon)

        # Train!
        print(">>>>>>>> Running training >>>>>>>>")
        print("  Num examples = %d" %(len(self.train_loader)*self.config.batch_size))
        print("  Num Epochs = %d" %self.config.epoch)
        print("  Instantaneous batch size per GPU = %d"%self.config.batch_size)
        print("  GPU ids = %s" %self.config.visible_device)
        print("  Total step = %d" %t_total)
        print("  Warm up step = %d" %warmup_steps)
        print("  FP16 Option = %s" %self.config.fp16)
        print("  Adv Option = %s" %self.config.adv_option)
        print(">>>>>>>> Running training >>>>>>>>")

        print(">>>>>>>> Model Structure >>>>>>>>")
        for name,parameters in self.model.named_parameters():
            print(name,':',parameters.size())
        print(">>>>>>>> Model Structure >>>>>>>>\n")

        print('start training..')
        writer = SummaryWriter(self.config.path_tensorboard)
        global_step = 0
        f1_best = 0
        # progress_bar = tqdm(range(len(train_loader)*self.config.epoch))
        for epo in range(self.config.epoch):
            progress_bar = ProgressBar(n_total=len(self.train_loader), desc='Training epoch:{0}'.format(epo))
            for i, bs in enumerate(self.train_loader):
                # 启动训练模式
                self.model.train()
                loss = self.step(bs)
                progress_bar(i, {'loss': loss.item()})
                global_step += 1     
                writer.add_scalar('loss', loss, global_step=global_step, walltime=None)
                # 模型保存
                if global_step % self.config.step_save==0 and global_step>0:
                    # 模型评估
                    f1_eval = eval(self.valid_loader, self.model, self.index2tgt, self.func_index2token)
                    # 模型保存
                    f1_best = self.save_checkpoint(global_step, f1_eval, f1_best)
            # 验证效果
            print('\nTest set Eval ' + '-'*20)
            eval(self.test_loader, self.model, self.index2tgt, self.func_index2token)
            # print('Test set: loss:%.6f  valid f1:%.3f ' %(epo, self.config.epoch, loss.item(), f1))
        print('training end..')
        writer.close()


    def step(self, bs):
        """
        每一个batch的训练过程/步骤
        """
        # 输入
        input_ids = bs[0]
        attention_mask = bs[1]
        labels = bs[2]
        
        # 定义loss，并训练
        outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)   #
        loss = outputs[0]           # 获取每个token的logit输出结果
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
            
        # 反向传播
        if self.config.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
            
        # 对抗训练
        if self.config.adv_option == 'FGM':
            self.fgm.attack()
            loss_adv = self.model(input_ids, labels=labels, attention_mask=attention_mask)[0]
            if torch.cuda.device_count() > 1:
                loss_adv = loss_adv.mean()
            loss_adv.backward()
            self.fgm.restore()
        if self.config.adv_option == 'PGD':
            self.pgd.backup_grad()
            K = 3
            for t in range(K):
                self.pgd.attack(is_first_attack=(t==0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    self.model.zero_grad()
                else:
                    self.pgd.restore_grad()
                loss_adv = self.model(input_ids, labels=labels, attention_mask=attention_mask)[0]
                loss_adv.backward()                      # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            self.pgd.restore()                           # 恢复embedding参数
        
        # 梯度操作
        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()
        # loss.backward()             
        # optimizer.step()
        # scheduler.step()
        # model.zero_grad()
        # # optimizer.zero_grad()   # 梯度清零
        return loss


    def save_checkpoint(self, step_current, f1_eval, f1_best):
        """
        模型保存
        """
        if f1_eval != 0:
            # 保存路径
            path = self.config.path_model + 'step_{}/'.format(step_current)
            if not os.path.exists(path):
                os.makedirs(path)
            # 保存当前step的模型
            if self.config.model_name in self.config.model_list_nopretrain:
                torch.save(self.model.state_dict(), self.config.path_model+'pytorch_model.bin')
            else:
                model_save = self.model.module if torch.cuda.device_count() > 1 else self.model
                model_save.save_pretrained(path)
            print('Saving model: {}'.format(path))
            if not os.path.exists(self.config.path_optimizer):
                os.mkdir(self.config.path_optimizer)
            torch.save(self.optimizer.state_dict(), os.path.join(self.config.path_optimizer, "optimizer.pt"))
            torch.save(self.scheduler.state_dict(), os.path.join(self.config.path_optimizer, "scheduler.pt"))
            print("Saving optimizer and scheduler states to %s"%self.config.path_model)
            # 保存最优的模型
            if f1_eval > f1_best:
                # 创建文件夹
                path_best = self.config.path_model + 'step_best/'
                if not os.path.exists(path_best):
                    os.makedirs(path_best)
                # 模型保存
                if self.config.model_name in self.config.model_list_nopretrain:
                    torch.save(self.model.state_dict(), path_best+'pytorch_model.bin')
                else:
                    model_save = self.model.module if torch.cuda.device_count() > 1 else self.model
                    model_save.save_pretrained(path_best)
                f1_best = f1_eval
                print('Saving best model: {}\n'.format(path_best))
        return f1_best
