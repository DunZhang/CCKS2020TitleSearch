import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TitleCLFDataSet(Dataset):
    def __init__(self, file_path: str, tokenizer: BertTokenizer, label2id: dict = None, max_len: int = 128,
                 has_label: bool = True):
        self.tokenizer = tokenizer
        if label2id:
            self.label2id = label2id
        else:
            # 默认的关系
            self.label2id = {"medical": 0, "book": 1}
        self.max_len = max_len
        self.has_label = has_label  # 数据是否有label，没有label就是真正的测试数据
        sens, labels = [], []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.split("\t")
                if has_label:
                    if len(ss) != 2:
                        continue
                    sen, label = ss[0].strip(), ss[1].strip()
                    sens.append(sen)
                    labels.append(label)
                else:
                    if len(ss) != 1:
                        continue
                    sen = ss[0].strip()
                    sens.append(sen)

        res = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens,
                                          add_special_tokens=True,
                                          pad_to_max_length=True,
                                          return_tensors="pt",
                                          max_length=self.max_len)

        self.input_ids, self.attention_mask, self.token_type_ids = res["input_ids"], res["attention_mask"], res[
            "token_type_ids"]
        if has_label:
            labels = [self.label2id[i] for i in labels]
            self.labels = torch.tensor(labels)
            print(self.labels)
        print(self.input_ids[0])
        print(self.attention_mask[0])
        print(self.token_type_ids[0])

    def __len__(self):  # 返回整个数据集的大小
        return self.input_ids.shape[0]

    def __getitem__(self, idx):  # 根据索引index返回dataset[index]
        if self.has_label:
            return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.labels[idx]
        else:
            return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx]


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(r"D:\谷歌下载目录\RoBERTa-tiny3L312-clue-pt")
    label2id = {"未知": 0, "引导": 1, "直回": 2}
    file_path = r"D:\Codes\PythonProj\FAQ_DL\test.txt"

    fd = TitleCLFDataSet(file_path, tokenizer, label2id, 12)
    # tokenizer.batch_encode_plus()
    # res = tokenizer.encode_plus(text="你好啊",
    #                             text_pair="你好吗啊",
    #                             add_special_tokens=True,
    #                             pad_to_max_length=True,
    #                             return_tensors="pt",
    #                             max_length=12)
    # print(res)
