
# -*- coding: UTF-8 -*-


import os
import time
import numpy as np
import torch
import logging
from arg_config import Config
from DataManager import DataManager
from process.train import train
from process.eval import eval
from process.test import test



if __name__ == '__main__':

    # 配置文件
    # logging.basicConfig(filename='./sys.log', filemode='a', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  #日志配置
    Config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = Config.visible_device

    # 设置随机种子，保证结果每次结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    start_time = time.time()

    # 数据处理
    print('read data...')
    dm = DataManager()
    # 获取参数
    num_labels = dm.num_labels
    tgt2index = dm.tgt2index
    index2tgt = dm.index2tgt

    # 模式
    if Config.mode == 'train':
        # 获取数据
        print('data process...')
        train_loader = dm.get_data_train()
        valid_loader = dm.get_data_valid()
        # 训练
        train(train_loader, valid_loader, num_labels, index2tgt)
    elif Config.mode == 'eval':
        # 评估
        valid_loader = dm.get_data_valid()
        # eval(valid_loader, model, id2label, func_index2token)
        pass
    elif Config.mode == 'test':
        # 测试
        test_loader = dm.get_data_test()
        test(Config, test_loader, index2tgt)
    else:
        print("no task going on!")
        print("you can use one of the following lists to replace the valible of Config.py. ['train', 'test', 'valid'] !")
        