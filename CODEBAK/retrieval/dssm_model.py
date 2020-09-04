import os
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import torch
from transformers import BertConfig
import torch.nn.functional as F


class RetrievalDSSM(torch.nn.Module):
    def __init__(self, bert_path_or_config):
        super(RetrievalDSSM, self).__init__()
        if isinstance(bert_path_or_config, str):
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path_or_config)
        elif isinstance(bert_path_or_config, BertConfig):
            self.bert = BertModel(bert_path_or_config)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path_or_config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        _, pooler_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
        return pooler_output

    def save(self, save_dir):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
