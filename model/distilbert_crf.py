import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.crf import CRF
from transformers import DistilBertPreTrainedModel, DistilBertModel



class DistilBertCRF(DistilBertPreTrainedModel):
    def __init__(self, config):
        super(DistilBertCRF, self).__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None,labels=None,input_lens=None):
        outputs =self.distilbert(input_ids = input_ids,attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # outputs = (logits,)
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions = logits, tags=labels, mask=attention_mask)
            # outputs = (logits,)
            # outputs =(loss,)+outputs
        outputs = (loss, logits)
        return outputs # (loss), scores
    
    