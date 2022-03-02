
import random

class Config(object):

    def __init__(self):
        
        self.mode = 'train'                                             # train/test
        self.continue_train = False                                     # 是否继续训练上次的模型
        self.visible_device = '0'                                       # 可见的GPU卡号
        self.device = 'cuda:0'                                          # 主卡号
        self.model_name = 'bert_crf'                                    # 模型名称，可选lstm_crf/transformer/albert_crf/bert_crf/distilbert_crf/roberta_crf
        self.language = 'zh'                                            # 数据源语言：zh-中文，en-英文

        self.model_list_nopretrain = ['lstm_crf', 'transformer']
        self.model_list_pretrain = ['albert_crf', 'bert_crf', 'distilbert_crf', 'roberta_crf']
        self.model_pretrain_online_checkpoint = 'bert-base-chinese'     #'albert-base-v2'
        self.model_pretrain_trainable = True

        # data
        self.dataset = 'CNER'                                           # 数据集名称
        self.path_dataset = './dataset/'+ self.dataset +'/'             # 数据集路径
        self.path_output = './dataset/'+ self.dataset +'/output.txt'    # Preditor输出路径

        # base model
        self.path_base = './checkpoints/' + self.dataset                # 保存路径
        self.path_tokenizer = self.path_base + '/tokenizer/'            # 分词器保存路径
        self.path_model = self.path_base + '/model/'                    # 模型保存路径
        self.path_optimizer = self.path_base + '/optimizer/'            # 优化器保存路径
        
        # model parameter
        self.learning_rate = 3e-5                                       # 模型学习率
        self.crf_learning_rate = 1e-3                                   # CRF层学习率
        self.warmup_proportion = 0.1                                    # warm up的步数比例x（全局总训练次数t中前x*t步）
        self.weight_decay = 0.01                                        # 权重衰减的比例
        self.adam_epsilon = 1e-08               

        # training
        self.epoch = 100                                                # 训练轮数
        self.step_save = 100                                            # 每间多少轮保存一次模型
        self.batch_size = 64                                            # 批次大小
        self.max_seq_length = 256                                       # 句子最长长度
        self.path_tgt_map = './dataset/'+ self.dataset +'/map.txt'      # 标签文件
        self.path_vocab = './dataset/'+ self.dataset +'/vocab.pkl'      # 词表
        self.path_tensorboard = './logs/tensorboard/'

        # 多卡训练配置
        # self.init_method = 'tcp://localhost:21339'
        self.port = str(random.randint(10000,60000))                    # 多卡训练进程间通讯端口
        self.init_method = 'tcp://localhost:' + self.port               # 多卡训练的通讯地址


        # 对抗训练
        self.adv_option = 'None'                                        # 是否引入对抗训练：none/FGM/PGD
        self.adv_name = 'word_embeddings'                               # 对抗训练的干扰位置
        self.adv_epsilon = 1.0                                          # epsilon参数
        # 混合精度训练
        self.fp16 = False                                               # 是否开启混合精度训练
        self.fp16_opt_level = 'O1'                                      # 训练可选'O1'，测试可选'O3'
        


        # pretrain model
        self.pretrain_model = [
            'bert-base-chinese',                                # BERT: Google开源中文预训练模型（12-layer, 768-hidden, 12-heads, 110M parameters）
            'hfl/chinese-bert-wwm',                             # BERT: 中文wiki数据训练的Whole Word Mask版本（12-layer, 768-hidden, 12-heads, 110M parameters）
            'hfl/chinese-bert-wwm-ext',                         # BERT: 使用额外数据训练的Whole Word Mask版本（12-layer, 768-hidden, 12-heads, 110M parameters）
            'hfl/chinese-roberta-wwm-ext',                      # RoBERTa: 使用额外数据训练的Whole Word Mask版本（12-layer, 768-hidden, 12-heads, 110M parameters）
            'hfl/chinese-roberta-wwm-ext-large',                # RoBERTa: 使用额外数据训练的Whole Word Mask+Large 版本（24-layer, 1024-hidden, 16-heads, 330M parameters）
            'voidful/albert_chinese_base',                      # Albert: 非官方+base版（12layer）
            'voidful/albert_chinese_large',                     # Albert: 非官方+large版（24layer）
            'hfl/chinese-electra-base-discriminator',           # ELECTRA: 中文版+discriminator（12层，隐层768，12个注意力头，学习率2e-4，batch256，最大长度512，训练1M步）
            'hfl/chinese-electra-large-discriminator',          # ELECTRA: 中文版+discriminator+large （24层，隐层1024，16个注意力头，学习率1e-4，batch96，最大长度512，训练2M步）
            'hfl/chinese-electra-180g-base-discriminator',      # ELECTRA: 中文版+discriminator+大训练语料（12层，隐层768，12个注意力头，学习率2e-4，batch256，最大长度512，训练1M步）
            'hfl/chinese-electra-180g-large-discriminator',     # ELECTRA: 中文版+discriminator+大训练语料+large （24层，隐层1024，16个注意力头，学习率1e-4，batch96，最大长度512，训练2M步）
        ]
        
        
        ## 中文模型出处：
        # bert: https://github.com/huggingface/transformers
        # bert-wwm/roberta: https://github.com/ymcui/Chinese-BERT-wwm
        # albert: https://github.com/brightmart/albert_zh
        # electra: https://github.com/ymcui/Chinese-ELECTRA

