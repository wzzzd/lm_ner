

from transformers import BertTokenizer, AlbertTokenizer
from model.bert_crf import BertCRF
from model.roberta_crf import RoBertaCRF
from model.lstm_crf import LSTM_CRF
from model.transformer_crf import TransformerCRF
from model.albert_crf import AlbertCRF
from model.distilbert_crf import DistilBertCRF



def map_model(name):
    """模型映射"""
    # if name == 'lstm_crf':
    #     model = LSTM_CRF
    # elif name == 'transformer':
    #     model = TransformerCRF
    # elif name == 'bert_crf':
    #     model = BertCRF
    # elif name == 'roberta_crf':
    #     model = RoBertaCRF
    # elif name == 'albert_crf':
    #     model = AlbertCRF
    # else:
    #     model = BertCRF
    map_func = {
        'lstm_crf' : LSTM_CRF,
        'transformer' : TransformerCRF,
        'bert_crf' : BertCRF,
        'roberta_crf' : RoBertaCRF,
        'albert_crf' : AlbertCRF,
        'distilbert_crf' : DistilBertCRF
    }
    model = map_func.get(name, BertCRF)
    return model
    
        
        
        
def map_tokenizer(name):
    """模型映射"""
    if name == 'albert_crf':
        tokenizer = AlbertTokenizer
    else:
        tokenizer = BertTokenizer
    return tokenizer
        