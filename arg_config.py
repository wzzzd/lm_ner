

class Config(object):

    def __init__(self):
        
        self.mode = 'train'
        self.device = 'cuda:6'
        self.model_name = 'bert-base-uncased'#'bert-base-chinese'    #bert-base-uncased
        self.language = 'en'

        # base model
        self.path_tokenizer = './checkpoints/tokenizer/'
        self.path_bert = './checkpoints/'
        self.path_model = './checkpoints/model/'
        self.path_optimizer = './checkpoints/optimizer/'
        
        # model parameter
        self.learning_rate = 3e-4
        self.crf_learning_rate = 1e-3
        self.warmup_proportion = 0.1            # warm up的步数比例x（全局总训练次数t中前x*t步）
        self.weight_decay = 0.01                # 权重衰减的比例
        self.adam_epsilon = 1e-08

        # data
        self.dataset = 'Shopline'
        self.path_dataset = './datasets/'+ self.dataset +'/'
        self.path_output = './datasets/'+ self.dataset +'/output.txt'

        # training
        self.epoch = 100
        self.batch_size = 32
        self.max_seq_length = 256
        self.path_save_model = './checkpoints/bert_crf/'
        self.path_tgt_map = './datasets/'+ self.dataset +'/map.txt'
        self.path_vocab = './datasets/'+ self.dataset +'/vocab.pkl'

        # tag type
        # self.tag_type = ['ORG', 'RACE', 'PRO', 'NAME', 'EDU', 'CONT', 'LOC', 'TITLE']
        # self.tag_method = 'BMES'     # B IO
        self.tag_type = ['ORG', 'LOC']
        self.tag_method = 'BIO'     # BIO


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
        
        ## 模型出处：
        # bert: https://github.com/huggingface/transformers
        # bert-wwm/roberta: https://github.com/ymcui/Chinese-BERT-wwm
        # albert: https://github.com/brightmart/albert_zh
        # electra: https://github.com/ymcui/Chinese-ELECTRA

