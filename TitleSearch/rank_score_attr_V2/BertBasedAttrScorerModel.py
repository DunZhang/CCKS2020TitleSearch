import os
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import torch
from transformers import BertConfig
import torch.nn.functional as F
import time


class BertBasedAttrScorerModel(torch.nn.Module):
    def __init__(self, bert_path_or_config, fc_path=None):
        super(BertBasedAttrScorerModel, self).__init__()
        if isinstance(bert_path_or_config, str):
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path_or_config)
            self.tokenizer = BertTokenizer.from_pretrained(bert_path_or_config)
        elif isinstance(bert_path_or_config, BertConfig):
            self.bert = BertModel(bert_path_or_config)
            self.tokenizer = None

        self.fc = torch.nn.Linear(in_features=self.bert.config.hidden_size, out_features=1, bias=True)
        if fc_path:
            self.fc.load_state_dict(torch.load(fc_path))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, null_val_mask=None):
        _, pooler_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
        scores = torch.nn.functional.relu(self.fc(pooler_output) * null_val_mask)
        return scores

    def save(self, save_dir):
        self.bert.save_pretrained(save_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)
        torch.save(self.fc.state_dict(), os.path.join(save_dir, "fc_weight.bin"))


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertBasedAttrScorerModel(r"G:\Data\roberta_tiny_clue").to(device)
    model.eval()
    torch.cuda.empty_cache()
    x, y, z = torch.ones(size=(300, 100)).to(device).long(), torch.ones(size=(300, 100)).to(
        device).long(), torch.zeros(size=(300, 100)).to(device).long()
    ss = time.time()
    with torch.no_grad():
        for _ in range(1):
            s = model(x, y, z)
            print(s.shape)
    ee = time.time()
    print(ee - ss)
