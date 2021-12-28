

import torch
from torch import nn
# from torchcrf import CRF
from model.layers.crf import CRF
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertConfig, BertTokenizer, BertModel, BertPreTrainedModel, BertForSequenceClassification
from transformers import AutoModel, AutoTokenizer, PreTrainedModel


class RoBertaCRF(BertPreTrainedModel):
    def __init__(self, config):
        super(RoBertaCRF, self).__init__(config)
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # outputs = (logits,)
        loss = None
        if labels is not None:
            size = logits.size()
            labels = labels[:,:size[1]]
            loss = -1 * self.crf(emissions = logits, tags=labels, mask=attention_mask)

            # crf decode
            logits = self.crf.decode(emissions = logits)    # , mask=attention_mask
            # logits = logits.transpose(0, 1)
            logits = logits.permute(1,2,0)
            # # logits = torch.squeeze(logits, dim=0)
            # outputs = (logits,)
            # outputs =(-1*loss,)+outputs
        outputs = (loss, logits)

        return outputs # (loss), scores
