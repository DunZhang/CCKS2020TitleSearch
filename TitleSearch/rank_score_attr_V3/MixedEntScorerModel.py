"""
第二代scorer model
attr scorer用bert或者手工特征+线性回归
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
from ManualFeatureBasedAttrScoerModel import ManualFeatureBasedAttrScoerModel

import os
from os.path import join


class MixedEntScorerModel(torch.nn.Module):
    def __init__(self, model_dir, share_bert_weight=False, use_manual_feature=False):
        super(MixedEntScorerModel, self).__init__()
        self.manual_attr_names = ["company", "bases", "place", "spec"]
        self.bert_attr_names = ["name", "functions"]
        self.use_manual_feature = use_manual_feature
        self.modules = {}
        # 加载现有模型
        if all([os.path.exists(join(model_dir, "{}_attr_scoer".format(attr_name))) for attr_name in
                self.bert_attr_names]):
            print("加载微调后的模型")
            ### 用于手工特征的模型
            self.modules.update(
                {attr_name: ManualFeatureBasedAttrScoerModel() for attr_name in self.manual_attr_names})
            ### 加载 function he name的打分模型
            if share_bert_weight:
                sub_model = BertBasedAttrScorerModel(join(model_dir, "{}_attr_scoer".format("name")),
                                                     join(model_dir, "{}_attr_scoer/fc_weight.bin".format("name")))
                self.modules.update(
                    {attr_name: sub_model for attr_name in self.bert_attr_names})
            else:
                self.modules.update(
                    {attr_name: BertBasedAttrScorerModel(
                        join(model_dir, "{}_attr_scoer".format(attr_name)),
                        join(model_dir, "{}_attr_scoer/fc_weight.bin".format(attr_name))) for
                        attr_name in ["name", "functions"]})

        else:  # 创建新的模型
            ### 用于手工特征的模型
            self.modules.update(
                {attr_name: ManualFeatureBasedAttrScoerModel() for attr_name in self.manual_attr_names})
            ### 加载 function 和 name的打分模型
            if share_bert_weight:
                sub_model = BertBasedAttrScorerModel(model_dir)
                self.modules.update(
                    {attr_name: sub_model for attr_name in self.bert_attr_names})
            else:
                self.modules.update(
                    {attr_name: BertBasedAttrScorerModel(model_dir) for attr_name in self.bert_attr_names})
        for k, v in self.modules.items():
            self.add_module(k, v)

    def forward(self, ipts):
        name_score = self.modules["name"](**ipts["name"])
        functions_score = self.modules["functions"](**ipts["functions"])
        zeros = functions_score * 0
        company_score, bases_score, place_score, spec_score = zeros, zeros, zeros, zeros
        if self.use_manual_feature:
            company_score = self.modules["company"](**ipts["company"])
            bases_score = self.modules["bases"](**ipts["bases"])
            place_score = self.modules["place"](**ipts["place"])
            spec_score = self.modules["spec"](**ipts["spec"])

        total_score = name_score + functions_score + company_score + bases_score + place_score + spec_score
        return {"total_score": total_score, "name_score": name_score, "company_score": company_score,
                "bases_score": bases_score, "functions_score": functions_score, "place_score": place_score,
                "spec_score": spec_score}

    def save(self, save_dir):
        for name, model in self.modules.items():
            submodel_save_dir = os.path.join(save_dir, "{}_attr_scoer".format(name))
            if not os.path.exists(submodel_save_dir):
                os.makedirs(submodel_save_dir)
            model.save(submodel_save_dir)

        # 存储分数融合层
        if hasattr(self, "fc1"):
            torch.save(self.fc1.state_dict(), join(save_dir, "fc1_weight.bin"))
        if hasattr(self, "fc2"):
            torch.save(self.fc2.state_dict(), join(save_dir, "fc2_weight.bin"))


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixedEntScorerModel(r"G:\Data\roberta_tiny_clue", share_bert_weight=True).to(device)
    for n, p in model.named_parameters():
        print(n)
