"""
第一代scorer model
每个attr scorer是用的bert
"""
import os
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import torch
from transformers import BertConfig
import torch.nn.functional as F
import time
from BertBasedAttrScorerModel import BertBasedAttrScorerModel

import os


class EntScorerModel(torch.nn.Module):
    def __init__(self, model_dir, share_weight=False):
        super(EntScorerModel, self).__init__()
        self.attr_names = ["name", "company", "bases", "functions", "place"]
        # 加载现有模型
        if all([os.path.exists(os.path.join(model_dir, "{}_attr_scoer".format(attr_name))) for attr_name in
                self.attr_names]):
            print("加载微调后的模型")
            if share_weight:
                sub_model = BertBasedAttrScorerModel(os.path.join(model_dir, "{}_attr_scoer".format("bases")),
                                                     os.path.join(model_dir,
                                                                  "{}_attr_scoer/fc_weight.bin".format("bases")))
                self.modules = {attr_name: sub_model for attr_name in self.attr_names}
            else:
                self.modules = {
                    attr_name: BertBasedAttrScorerModel(os.path.join(model_dir, "{}_attr_scoer".format(attr_name)),
                                                        os.path.join(model_dir,
                                                                     "{}_attr_scoer/fc_weight.bin".format(attr_name)))
                    for attr_name in self.attr_names}
        elif share_weight:  # 共用一个模型
            attr_score_model = BertBasedAttrScorerModel(model_dir)
            self.modules = {attr_name: attr_score_model for attr_name in self.attr_names}
        else:  # 一个属性一个模型
            self.modules = {attr_name: BertBasedAttrScorerModel(model_dir) for attr_name in self.attr_names}

        for k, v in self.modules.items():
            self.add_module(k, v)

    def forward(self, ipts):
        name_score = self.modules["name"](**ipts["name"])
        company_score = self.modules["company"](**ipts["company"])
        bases_score = self.modules["bases"](**ipts["bases"])
        functions_score = self.modules["functions"](**ipts["functions"])
        place_score = self.modules["place"](**ipts["place"])
        total_score = name_score + company_score + bases_score + functions_score + place_score
        # total_score = name_score  + bases_score + functions_score
        return {"total_score": total_score, "name_score": name_score, "company_score": company_score,
                "bases_score": bases_score, "functions_score": functions_score, "place_score": place_score}

    def save(self, save_dir):
        for name, model in self.modules.items():
            submodel_save_dir = os.path.join(save_dir, "{}_attr_scoer".format(name))
            if not os.path.exists(submodel_save_dir):
                os.makedirs(submodel_save_dir)
            model.save(submodel_save_dir)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EntScorerModel(r"G:\Data\roberta_tiny_clue", share_weight=True).to(device)
    for n, p in model.named_parameters():
        print(n)
