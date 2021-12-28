
# -*- coding: UTF-8 -*-


import os
import time
import numpy as np
import torch
import logging
from Config import Config
from DataManager import DataManager
from process.Trainer import Trainer
from process.Predictor import Predictor

if __name__ == '__main__':

    # 配置文件
    # logging.basicConfig(filename='./sys.log', filemode='a', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  #日志配置
    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.visible_device

    # 设置随机种子，保证结果每次结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    start_time = time.time()

    # 数据处理
    print('read data...')
    dm = DataManager()
    
    # 模式
    if config.mode == 'train':
        # 获取数据
        print('data process...')
        train_loader = dm.get_data_train()
        valid_loader = dm.get_data_valid()
        test_loader = dm.get_data_test()
        # 训练
        trainer = Trainer(config, train_loader, valid_loader, test_loader)
        trainer.train()
    else:
        # 测试
        test_loader = dm.get_data_test()
        predictor = Predictor(config, test_loader)
        predictor.predict()

    
    