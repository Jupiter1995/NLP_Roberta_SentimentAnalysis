# import libraries
import torch
from torch import nn

from transformers import RobertaModel

# Construct the classifier for sentiment analysis
class SentimentCalssify(nn.Module):

    def __init__(self, dropout=0.5):

        super(SentimentCalssify, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')

        # Freeze all layers except the classifier (the last layer)
        for name, param in self.roberta.named_parameters():
            param.requires_grad = False
        self.l1 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(768, 3)
        

    def forward(self, input_id, mask):

        _, pooled_output = self.roberta(input_ids= input_id, attention_mask=mask,return_dict=False)
        pre_classify = self.l1(pooled_output)
        dropout_output = self.dropout(pre_classify)
        activate_output = self.activation(dropout_output)
        linear_output = self.linear(activate_output)

        return linear_output