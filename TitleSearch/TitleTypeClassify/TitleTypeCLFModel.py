import os
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import torch
from transformers import BertConfig
import torch.nn.functional as F


class TitleTypeCLFModel(torch.nn.Module):
    def __init__(self, bert_path_or_config, num_labels, fc_path=None):
        super(TitleTypeCLFModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path_or_config)
        self.fc = torch.nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_labels, bias=True)
        if fc_path:
            self.fc.load_state_dict(torch.load(fc_path))
        self.tokenizer = BertTokenizer.from_pretrained(bert_path_or_config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        sen_vec = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)[1]
        logits = self.fc(sen_vec)  # batch_size * num_class
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return logits, loss.mean()
        return logits, None

    def save(self, save_dir):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.fc.state_dict(), os.path.join(save_dir, "classifier.bin"))
