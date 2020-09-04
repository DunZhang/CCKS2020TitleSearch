from Evaluate import eval, get_pred_res
from Model import CLFModel
from transformers import BertTokenizer
from CLFDataSet import CLFDataSet
import json
from torch.utils.data import DataLoader, SequentialSampler
from os.path import join
import torch
import pandas as pd

if __name__ == "__main__":
    ### 相关全局变量
    model_dir = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\rank_score_attr\ClassifyContainFunction\rbt\best_model"
    num_labels = 2
    device = "0"
    dev_file_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\test_data.txt"
    seq_length = 50
    batch_size = 16
    ### 开始评测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    clf_model = CLFModel(model_dir, num_labels, join(model_dir, "fc_weight.bin")).to(device)
    clf_model.eval()

    dev_data_set = CLFDataSet(file_path=dev_file_path, tokenizer=tokenizer, max_len=seq_length, task="test")
    dev_data_loader = DataLoader(dataset=dev_data_set, batch_size=batch_size,
                                 sampler=SequentialSampler(dev_data_set))
    preds = get_pred_res(clf_model, dev_data_loader, device)

    data = []
    with open(dev_file_path, "r", encoding="utf8") as fr:
        for line in fr:
            data.append(line.strip())
    assert len(data) == len(preds)
    data = list(zip(data, preds))
    pd.DataFrame(data, columns=["Sen", "Res"]).to_excel("pred.xlsx", index=False)
