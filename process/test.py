

import re
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import AutoModel, AutoTokenizer, AutoConfig
from config.BertConfig import BertConfig
from model.lstm_crf import LSTM_CRF
from model.bert_crf import BertCRF
from process.eval import eval


def test(Config, test_loader, id2label, vocab=''):
    """
    测试
    """
    # 读取模型
    if Config.model_name in ['lstm_crf']:
        tokenizer = pkl.load(open(Config.path_vocab, 'rb'))
        index2token = {i:x for x,i in tokenizer.items()}
        func_index2token = lambda x: index2token[x]
        model = map_model(Config.model_name)
        model.load_state_dict(torch.load(Config.path_model))
    else:
        tokenizer = BertTokenizer.from_pretrained(Config.path_tokenizer)
        func_index2token = tokenizer.convert_ids_to_tokens
        model = map_model(Config.model_name)
        model = model.from_pretrained(Config.path_model)
    # inference
    model.eval()
    model.to(Config.device)
    # f1 = eval(config, model, test_iter, test=True, vocab=vocab)
    src, predict = infer(test_loader, model, id2label, func_index2token)
    predict = [ bio2token(s, pred, Config.tag_type, tokenizer, Config.model_name, Config.language) for s, pred in zip(src, predict) ]

    # 格式转换
    src = [src2token(s, tokenizer, Config.model_name, Config.language) for s in src]
    data = {
        'src':src,
        'predict':predict
    }
    data = pd.DataFrame(data, index=range(len(src)))
    data.to_csv(Config.path_output, sep='\t', index=False)
    print('output path: %s' %Config.path_output)
    return data


def map_model(name):
    """模型映射"""
    if name == 'lstm_crf':
        model = LSTM_CRF()
    elif name == 'transformer':
        model = TransformerCRF
    elif name == '':
        model = BertCRF()
    elif name == '':
        model = RoBertaCRF
    else:
        model = BertCRF
    return model
        



def infer(loader, model, id2label, func_index2token):
    
    list_src = []
    list_predict = []

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

        # 条件到数组
        list_src.extend(line_input)
        list_predict.extend(predicts)

    return list_src, list_predict


def bio2token(src, tgt, lab_class, tokenizer, model_name, lang):
    """将BIO标签转换成token"""
    tgt = [re.sub(r'B-|I-', '', x) for x in tgt]
    # tgt = [[x,i] for i, x in enumerate(tgt)]
    list_label = []
    for c in lab_class:
        tmp_tgt = [[x,i] for i, x in enumerate(tgt) if c in x]
        if len(tmp_tgt) == 1:
            start = tmp_tgt[0][1]
            tmp_lab = src2token(src[start], tokenizer, model_name, lang)
            list_label.append((c, tmp_lab, [start]))
        if len(tmp_tgt) > 1:
            start = tmp_tgt[0][1]
            end = tmp_tgt[-1][1]
            tmp_lab = src2token(src[start:end+1], tokenizer, model_name, lang)
            list_label.append((c, tmp_lab, [start, end]))
    return list_label


def src2token(src, tokenizer, model_name, lang='zh'):
    """将input转换成标准输出格式"""
    if lang == 'zh':
        src = ''.join(src)
    else:
        if model_name in ['lstm_crf']:
            src = ' '.join(src)
        else:
            src = tokenizer.convert_tokens_to_string(src)
    return src
