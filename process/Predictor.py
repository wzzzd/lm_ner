from model.albert_crf import AlbertCRF
import re
import torch
import pandas as pd
import pickle as pkl
from process.MapModel import map_model, map_tokenizer
from process.eval import eval


class Predictor(object):
    
    def __init__(self, config, test_loader, metric=True):
        self.config = config
        self.loader = test_loader
        self.metric = True
        # 加载标签
        self.id2label = { int(x.strip().split(' ')[1]):x.strip().split(' ')[0] for x in open(self.config.path_tgt_map,'r').readlines()}
        # 加载模型
        self.load_model()
        
            
    def load_model(self):
        """
        读取模型
        """
        # 读取模型
        if self.config.model_name in self.config.model_list_nopretrain:
            self.tokenizer = pkl.load(open(self.config.path_vocab, 'rb'))
            self.index2token = {i:x for x,i in self.tokenizer.items()}
            self.func_index2token = lambda x: self.index2token[x]
            # model = map_model(self.config.model_name)
            # model(self.config).load_state_dict(torch.load(self.config.path_model + 'pytorch_model.bin'))    # 
            # self.model = LSTM_CRF(self.config)
            self.model = map_model(self.config.model_name)(self.config)
            self.model.load_state_dict(torch.load(self.config.path_model + 'pytorch_model.bin'))
        else:
            self.tokenizer = map_tokenizer(self.config.model_name).from_pretrained(self.config.path_tokenizer)
            # tokenizer = BertTokenizer.from_pretrained(self.config.path_tokenizer)
            self.func_index2token = self.tokenizer.convert_ids_to_tokens
            self.model = map_model(self.config.model_name)
            self.model = self.model.from_pretrained(self.config.path_model + 'step_best')
        # inference
        self.model.eval()
        self.model.to(self.config.device)
    
    
    def predict(self):
        """
        预测
        """
        # 是否计算指标
        if self.metric:
            eval(self.loader, self.model, self.id2label, self.func_index2token)
        
        # 推断
        src, predict, label = self.infer()
        tag_type= [re.sub(r'B-|I-', '', x) for x in self.id2label.values() if x!='O']
        tag_type = list(set(tag_type))
        predict = [ self.bio2token(s, pred, tag_type) for s, pred in zip(src, predict) ]
        label = [ self.bio2token(s, pred, tag_type) for s, pred in zip(src, label) ]
        # 格式转换
        src = [ self.src2token(s) for s in src]
        data = {
            'src':src,
            'predict':predict,
            'label':label
        }
        data = pd.DataFrame(data, index=range(len(src)))
        data.to_csv(self.config.path_output, sep='\t', index=False)
        print('output path: %s' %self.config.path_output)
        return data


    def infer(self):
        """
        推断过程
        """
        list_src = []
        list_predict = []
        list_label = []

        # 遍历每个batch
        for bs in self.loader:
            # 输入
            input_ids = bs[0]
            att_mask = bs[1]
            # 输入index转文字
            line_input = [[self.func_index2token(x) for x in line  if self.func_index2token(x) != '[PAD]'] for line in input_ids.cpu().numpy().tolist() ]
            # 输出
            outputs = self.model(input_ids)
            outputs = outputs[1]
            outputs = self.model.crf.decode(outputs, att_mask)                       # (1, batch_size, seq_size)
            predicts = torch.squeeze(outputs,dim=0).cpu().numpy().tolist()      # (batch_size, seq_size)
            src_size = input_ids.size()[0]

            assert src_size == len(predicts), "valid set: length difference between tgt and pred, in batch:%s" %str(i)
            input_ids = input_ids.tolist()

            # index 转 label
            predicts = [[self.id2label[x] for x in pre] for pre in predicts]
            label = [[self.id2label[x] for x in pre] for pre in bs[2].cpu().numpy().tolist()]

            # 条件到数组
            list_src.extend(line_input)
            list_predict.extend(predicts)
            list_label.extend(label)

        return list_src, list_predict, list_label


    def bio2token(self, src, tgt, lab_class):
        """
        将BIO标签转换成token
        """
        tgt = [re.sub(r'B-|I-', '', x) for x in tgt]
        list_label = []
        for c in lab_class:
            tmp_tgt = [ '_'.join([x,str(i)]) if c == x else '##' for i, x in enumerate(tgt)]
            tmp_tgt = '|'.join(tmp_tgt)
            tmp_tgt = tmp_tgt.split('##')
            for ele in tmp_tgt:
                if ele in ['', '|']:
                    continue
                tmp_ele = [ x for x in ele.split('|') if x!='' ]
                if len(tmp_ele) == 1:
                    start = int(tmp_ele[0].split('_')[1])
                    tmp_lab = self.src2token(src[start], dim='tgt')
                    if tmp_lab != '':
                        list_label.append((c, tmp_lab, [start]))
                if len(tmp_ele) > 1:
                    start = int(tmp_ele[0].split('_')[1])
                    end = int(tmp_ele[-1].split('_')[1])
                    tmp_lab = self.src2token(src[start:end+1], dim='tgt')
                    if tmp_lab != '':
                        list_label.append((c, tmp_lab, [start, end]))
        return list_label


    def src2token(self, src, dim='src'):
        """将input转换成标准输出格式"""
        if self.config.language == 'zh':
            src = ''.join(src)
        else:
            if self.config.model_name in self.config.model_list_nopretrain:
                src = ' '.join(src)
            else:
                if dim != 'src':
                    sign = [self.tokenizer.unk_token,
                            self.tokenizer.mask_token,
                            self.tokenizer.cls_token,
                            self.tokenizer.pad_token]
                    src = [ x for x in src[1:-1] if x not in sign]
                else:
                    src = src[1:-1]
                src = self.tokenizer.convert_tokens_to_string(src)
        return src


