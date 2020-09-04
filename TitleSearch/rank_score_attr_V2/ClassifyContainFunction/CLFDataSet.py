import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class CLFDataSet(Dataset):
    def __init__(self, file_path: str, tokenizer: BertTokenizer, max_len: int = 128, task="train"):
        self.tokenizer = tokenizer
        self.label2id = {0: 0, 1: 1}
        self.task = task
        self.max_len = max_len
        sens, labels = [], []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.split("\t")
                if task == "train" or task == "dev":
                    if len(ss) != 2:
                        continue
                    sen, label = ss[0].strip(), int(ss[1].strip())

                    sens.append(sen)
                    # print(len(sen))
                    labels.append(label)
                else:
                    sens.append(ss[0].strip())

        res = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens,
                                          add_special_tokens=True,
                                          pad_to_max_length=True,
                                          return_tensors="pt",
                                          max_length=self.max_len)
        self.input_ids, self.attention_mask, self.token_type_ids = res["input_ids"], res["attention_mask"], res[
            "token_type_ids"]
        if task == "train" or task == "dev":
            labels = [self.label2id[i] for i in labels]
            self.labels = torch.tensor(labels)
        else:
            self.labels = None
        # print(self.input_ids[0])
        # print(self.attention_mask[0])
        # print(self.token_type_ids[0])
        # print(self.labels)

    def __len__(self):  # 返回整个数据集的大小
        return self.input_ids.shape[0]

    def __getitem__(self, idx):  # 根据索引index返回dataset[index]
        if self.task == "train" or self.task == "dev":
            return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.labels[idx]
        else:
            return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx]


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(r"G:\Data\RBT3")
    file_path = "train.txt"

    fd = CLFDataSet(file_path, tokenizer, 50)
    # tokenizer.batch_encode_plus()
    # res = tokenizer.encode_plus(text="你好啊",
    #                             text_pair="你好吗啊",
    #                             add_special_tokens=True,
    #                             pad_to_max_length=True,
    #                             return_tensors="pt",
    #                             max_length=12)
    # print(res)
