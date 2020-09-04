""" DataSet for retrieval using DSSM structure """
from torch.utils.data import Dataset
from transformers import BertTokenizer


class RetrievalDSSMDataSet(Dataset):
    def __init__(self, file_path: str, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer
        sens_uq, sens_entinfo = [], []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.split("\t")
                if len(ss) != 2:
                    continue
                sen_a, sen_b = ss[0].strip(), ss[1].strip()
                sens_uq.append(sen_a)
                sens_entinfo.append(sen_b)

        # get bert input for query
        res = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens_uq,
                                          add_special_tokens=True,
                                          pad_to_max_length=True,
                                          return_tensors="pt",
                                          max_length=32)
        self.uq_input_ids, self.uq_attention_mask, self.uq_token_type_ids = res["input_ids"], res["attention_mask"], \
                                                                            res["token_type_ids"]

        # get bert input for entinfo
        res = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens_uq,
                                          add_special_tokens=True,
                                          pad_to_max_length=True,
                                          return_tensors="pt",
                                          max_length=80)
        self.ent_input_ids, self.ent_attention_mask, self.ent_token_type_ids = res["input_ids"], res["attention_mask"], \
                                                                               res["token_type_ids"]

    def __len__(self):  # 返回整个数据集的大小
        return self.uq_input_ids.shape[0]

    def __getitem__(self, idx):  # 根据索引index返回dataset[index]
        return self.uq_input_ids[idx], self.uq_attention_mask[idx], self.uq_token_type_ids[idx], self.ent_input_ids[
            idx], self.ent_attention_mask[idx], self.ent_token_type_ids[idx]


if __name__ == "__main__":
    pass
