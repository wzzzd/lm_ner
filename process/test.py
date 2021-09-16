

from model.albert_crf import AlbertCRF
import re
import torch
import pandas as pd
import pickle as pkl
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AlbertTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from config.BertConfig import BertConfig
from model.lstm_crf import LSTM_CRF
from model.bert_crf import BertCRF
from model.roberta_crf import RoBertaCRF
from model.transformer_crf import TransformerCRF
from process.eval import eval


def test(Config, test_loader, id2label, vocab=''):
    """
    测试
    """
    # 读取模型
    if Config.model_name in Config.model_list_nopretrain:
        tokenizer = pkl.load(open(Config.path_vocab, 'rb'))
        index2token = {i:x for x,i in tokenizer.items()}
        func_index2token = lambda x: index2token[x]
        # model = map_model(Config.model_name)
        # model(Config).load_state_dict(torch.load(Config.path_model + 'pytorch_model.bin'))    # 
        model = LSTM_CRF(Config)
        model.load_state_dict(torch.load(Config.path_model + 'pytorch_model.bin'))
    else:
        tokenizer = map_tokenizer(Config.model_name).from_pretrained(Config.path_tokenizer)
        # tokenizer = BertTokenizer.from_pretrained(Config.path_tokenizer)
        func_index2token = tokenizer.convert_ids_to_tokens
        model = map_model(Config.model_name)
        model = model.from_pretrained(Config.path_model)
    # inference
    model.eval()
    model.to(Config.device)
    # f1 = eval(config, model, test_iter, test=True, vocab=vocab)
    src, predict, label = infer(test_loader, model, id2label, func_index2token)
    tag_type= [re.sub(r'B-|I-', '', x) for x in id2label.values() if x!='O']
    tag_type = list(set(tag_type))
    predict = [ bio2token(s, pred, tag_type, tokenizer, Config.model_name, Config.model_list_nopretrain, Config.language) for s, pred in zip(src, predict) ]
    label = [ bio2token(s, pred, tag_type, tokenizer, Config.model_name, Config.model_list_nopretrain, Config.language) for s, pred in zip(src, label) ]

    # 格式转换
    src = [src2token(s, tokenizer, Config.model_name, Config.model_list_nopretrain ,Config.language) for s in src]
    data = {
        'src':src,
        'predict':predict,
        'label':label
    }
    data = pd.DataFrame(data, index=range(len(src)))
    data.to_csv(Config.path_output, sep='\t', index=False)
    print('output path: %s' %Config.path_output)
    return data


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
        



def infer(loader, model, id2label, func_index2token):
    
    list_src = []
    list_predict = []
    list_label = []

    # 遍历每个batch
    for bs in loader:
        # 输入
        input_ids = bs[0]
        att_mask = bs[1]
        # 输入index转文字
        line_input = [[func_index2token(x) for x in line  if func_index2token(x) != '[PAD]'] for line in input_ids.cpu().numpy().tolist() ]
        # 输出
        outputs = model(input_ids)
        outputs = outputs[1]
        outputs = model.crf.decode(outputs, att_mask)                       # (1, batch_size, seq_size)
        predicts = torch.squeeze(outputs,dim=0).cpu().numpy().tolist()      # (batch_size, seq_size)
        src_size = input_ids.size()[0]
        
        assert src_size == len(predicts), "valid set: length difference between tgt and pred, in batch:%s" %str(i)
        input_ids = input_ids.tolist()

        # index 转 label
        predicts = [[id2label[x] for x in pre] for pre in predicts]
        label = [[id2label[x] for x in pre] for pre in bs[2].cpu().numpy().tolist()]

        # 条件到数组
        list_src.extend(line_input)
        list_predict.extend(predicts)
        list_label.extend(label)

        
        ## line_input = [[func_index2token(x) for x in line  if func_index2token(x) != '[PAD]'] for line in input_ids.cpu().numpy().tolist() ]
        # from process.test import src2token
        # from arg_config import Config as Configs
        # from transformers import AlbertTokenizer
        # Config = Configs()
        # tokenizer = AlbertTokenizer.from_pretrained(Config.model_pretrain_online_checkpoint)
        # src = [src2token(s, tokenizer, Config.model_name, Config.model_list_nopretrain ,Config.language) for s in line_input]
        # label = [[id2label[x] for x in pre] for pre in bs[2].cpu().numpy().tolist()]
        # for i in range(len(src)):
        #     print('-'*50)
        #     print(src[i])
        #     print(label[i])
        #     print(predicts[i])


    return list_src, list_predict, list_label



def bio2token(src, tgt, lab_class, tokenizer, model_name, model_list_nopretrain, lang):
    """将BIO标签转换成token"""
    tgt = [re.sub(r'B-|I-', '', x) for x in tgt]
    # tgt = [[x,i] for i, x in enumerate(tgt)]
    list_label = []
    for c in lab_class:
        tmp_tgt = [ '_'.join([x,str(i)]) if c == x else '##' for i, x in enumerate(tgt)]
        # print('-'*100)
        # print(tmp_tgt)
        tmp_tgt = '|'.join(tmp_tgt)
        tmp_tgt = tmp_tgt.split('##')
        # print('-'*100)
        # print(tmp_tgt)
        for ele in tmp_tgt:
            if ele in ['', '|']:
                continue
            tmp_ele = [ x for x in ele.split('|') if x!='' ]
            if len(tmp_ele) == 1:
                start = int(tmp_ele[0].split('_')[1])
                tmp_lab = src2token(src[start], tokenizer, model_name, model_list_nopretrain, lang, dim='tgt')
                if tmp_lab != '':
                    list_label.append((c, tmp_lab, [start]))
            if len(tmp_ele) > 1:
                start = int(tmp_ele[0].split('_')[1])
                end = int(tmp_ele[-1].split('_')[1])
                tmp_lab = src2token(src[start:end+1], tokenizer, model_name, model_list_nopretrain, lang, dim='tgt')
                if tmp_lab != '':
                    list_label.append((c, tmp_lab, [start, end]))
    return list_label



# def bio2token(src, tgt, lab_class, tokenizer, model_name, lang):
#     """将BIO标签转换成token"""
#     tgt = [re.sub(r'B-|I-', '', x) for x in tgt]
#     entity = []
#     tmp_entity = []
#     tmp_index = []
#     for i, (word, lab) in enumerate(zip(src, tgt)):
#         if lab != 'O':
#             if tmp_entity == []:
#                 tmp_entity.append(word)
#                 tmp_index.append(i)
#             else:
#     return 


# def bio2token(src, tgt, lab_class, tokenizer, model_name, lang):
#     """将BIO标签转换成token"""
#     tgt = [re.sub(r'B-|I-', '', x) for x in tgt]
#     # tgt = [[x,i] for i, x in enumerate(tgt)]
#     list_label = []
#     for c in lab_class:
#         tmp_tgt = [[x,i] for i, x in enumerate(tgt) if c in x]
#         if len(tmp_tgt) == 1:
#             start = tmp_tgt[0][1]
#             tmp_lab = src2token(src[start], tokenizer, model_name, lang)
#             list_label.append((c, tmp_lab, [start]))
#         if len(tmp_tgt) > 1:
#             start = tmp_tgt[0][1]
#             end = tmp_tgt[-1][1]
#             tmp_lab = src2token(src[start:end+1], tokenizer, model_name, lang)
#             list_label.append((c, tmp_lab, [start, end]))
#     return list_label


def src2token(src, tokenizer, model_name, model_list_nopretrain, lang='zh', dim='src'):
    """将input转换成标准输出格式"""
    if lang == 'zh':
        src = ''.join(src)
    else:
        if model_name in model_list_nopretrain:
            src = ' '.join(src)
        else:
            if dim != 'src':
                sign = [tokenizer.unk_token,tokenizer.mask_token,tokenizer.cls_token,tokenizer.pad_token]
                src = [ x for x in src[1:-1] if x not in sign]
            else:
                src = src[1:-1]
            src = tokenizer.convert_tokens_to_string(src)
    return src


def map_tokenizer(name):
    """模型映射"""
    if name == 'albert_crf':
        tokenizer = AlbertTokenizer
    else:
        tokenizer = BertTokenizer
    return tokenizer
        
        