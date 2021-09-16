

import torch
from torch import nn
import pickle as pkl
# from torchcrf import CRF
from model.layers.crf import CRF
from torch.nn import CrossEntropyLoss, MSELoss
# from transformers import BertTokenizer, BertModel, BertPreTrainedModel
# from transformers.configuration_bert import BertConfig
# from transformers.modeling_bert import BertPreTrainedModel
from transformers import BertConfig, BertTokenizer, BertModel, BertPreTrainedModel, BertForSequenceClassification







class LSTM_CRF(nn.Module):
    def __init__(self, config):
        super(LSTM_CRF, self).__init__()
        self.emb_size = 300
        self.layer_num = 2
        self.hidden_size = 256
        self.num_label = len([ x for x in open(config.path_tgt_map, 'r').readlines() if x.strip()])
        self.vocab_size = len(pkl.load(open(config.path_vocab, 'rb')))
        self.dropout = nn.Dropout(0.3)

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, bidirectional=True, batch_first=True) #

        self.classifier = nn.Linear(self.hidden_size * 2, self.num_label)   # 
        self.crf = CRF(num_tags=self.num_label, batch_first=True)
 

    def forward(self, input_ids, labels=None, attention_mask=None):
        self.lstm.flatten_parameters()
        # embedding
        embeds = self.embedding(input_ids)
        # embeds = embeds.unsqueeze(1)
        # embeds = embeds.transpose(0,1)
        embeds = self.dropout(embeds)
        # lstm
        # print('emb size:', embeds.size())
        lstm_out, _ = self.lstm(embeds)

        # lstm_out = lstm_out.view(input_ids.size()[1], self.hidden_size*2)
        lstm_out = self.dropout(lstm_out)
        # full connect
        logits = self.classifier(lstm_out)
        output = (None, logits)
        # loss
        if labels is not None:
            size = logits.size()
            labels = labels[:,:size[1]]
            loss = -1*self.crf(emissions = logits, tags=labels, mask=attention_mask)
            output = (loss, logits)

        return output

