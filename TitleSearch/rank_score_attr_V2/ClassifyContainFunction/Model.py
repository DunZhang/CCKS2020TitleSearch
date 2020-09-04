import os
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import torch
from transformers import BertConfig
import torch.nn.functional as F


class CLFModel(torch.nn.Module):
    def __init__(self, bert_path_or_config, num_labels, fc_path=None):
        super(CLFModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path_or_config)
        self.fc = torch.nn.Linear(in_features=self.bert.config.hidden_size,
                                  out_features=num_labels, bias=True)
        if fc_path:
            self.fc.load_state_dict(torch.load(fc_path))
        self.tokenizer = BertTokenizer.from_pretrained(bert_path_or_config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        _, pooler_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
        logits = self.fc(pooler_output)  # batch_size * num_class
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return logits, loss.mean()
        return logits, 0.0

    def save(self, save_dir):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.fc.state_dict(), os.path.join(save_dir, "fc_weight.bin"))

    def get_sens_vec(self, sens: list):
        self.bert.eval()
        res = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens, pad_to_max_length=True,
                                               return_tensors="pt")
        input_ids = res["input_ids"]
        attention_mask = res["attention_mask"]
        token_type_ids = res["token_type_ids"]

        tensor_dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
        sampler = SequentialSampler(tensor_dataset)
        data_loader = DataLoader(tensor_dataset, sampler=sampler, batch_size=256)
        ### get sen vec
        all_sen_vec = []
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = [i.to(self.device) for i in batch_data]
                token_embeddings, pooler_output = self.bert(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                            token_type_ids=batch_data[2])
                sen_vecs = []
                for pooling_mode in self.pooling_modes:
                    if pooling_mode == "cls":
                        sen_vec = pooler_output
                    elif pooling_mode == "mean":
                        # get mean token sen vec
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = input_mask_expanded.sum(1)
                        sum_mask = torch.clamp(sum_mask, min=1e-9)
                        sen_vec = sum_embeddings / sum_mask
                    elif pooling_mode == "max":
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                        sen_vec = torch.max(token_embeddings, 1)[0]
                    sen_vecs.append(sen_vec)
                sen_vec = torch.cat(sen_vecs, 1)

                all_sen_vec.append(sen_vec.to("cpu").numpy())
        self.bert.train()
        return np.vstack(all_sen_vec)
