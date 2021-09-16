

import torch
from torch import nn
# from torchcrf import CRF
from model.layers.crf import CRF
from torch.nn import CrossEntropyLoss, MSELoss
# from transformers import BertTokenizer, BertModel, BertPreTrainedModel
# from transformers.configuration_bert import BertConfig
# from transformers.modeling_bert import BertPreTrainedModel
from transformers import BertConfig, BertTokenizer, BertModel, BertPreTrainedModel, BertForSequenceClassification



class BertCRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCRF, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        
        # from functools import reduce
        # hidden_layer = outputs.hidden_states
        # avg_hidden_layer = reduce(lambda x,y: x+y, hidden_layer)
        # sequence_output = avg_hidden_layer/float(len(hidden_layer))
        # pooled_output = self.pooler(avg_hidden_layer)
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (None, logits)
        if labels is not None:
            size = logits.size()
            labels = labels[:,:size[1]]
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,) + outputs
            # print(loss)

            # crf decode
            # logits = self.crf.decode(emissions = logits)    # , mask=attention_mask
            # # logits = logits.transpose(0, 1)
            # logits = logits.permute(1,2,0)
            # # # logits = torch.squeeze(logits, dim=0)
            # outputs = (logits,)
            # outputs =(-1*loss,)+outputs
        return outputs # (loss), scores


# class BertCrf(BertPreTrainedModel):
#     """
#         Bert + CRF
#     """
#     def __init__(self, config):
#         super(BertCrf, self).__init__(config)
#         self.bert = BertModel(config)                                       # Bert
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)               # dropout
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # linear layer
#         self.crf = CRF(num_tags=config.num_labels, batch_first=True)        # CRF
#         self.num_labels = config.num_labels
#         self.init_weights()
#         # self.fc = nn.Linear(arg_config.bert_hidden_size, arg_config.num_labels)  

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
#         outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
#         sequence_output = outputs[0]                                        # Bert的每个token输出的logit
#         sequence_output = self.dropout(sequence_output)                     
#         logits = self.classifier(sequence_output)                           # linear layer
#         outputs = (logits,)

#         # Loss
#         # softmax
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             # 截取labels长度
#             seq_len = logits.size()[1]
#             labels = labels[:,:seq_len].contiguous()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs =(loss,)+outputs

#         # # crf
#         # if labels is not None:
#         #     # logits = logits.transpose(0, 1)      # (seq_len, batch_size, num_labels)
#         #     # labels = labels.transpose(0, 1)
#         #     logits = logits[:,1:-1,:]
#         #     size = logits.size()
#         #     labels = labels[:,:size[1]]
#         #     if attention_mask is not None:
#         #         attention_mask = attention_mask[:,1:-1]#.uint8()            # 去除[CLS]、[SEP]
#         #         attention_mask = attention_mask.to(torch.uint8)
#         #         # attention_mask = attention_mask.transpose(0, 1)
#         #         loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
#         #     else:
#         #         loss = self.crf(emissions = logits, tags=labels)
#         #     outputs =(-1*loss,)+outputs

        
#         # paper loss （后续修改）
#         # pass
#         return outputs # (loss), scores
