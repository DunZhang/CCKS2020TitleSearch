import json
from FeatureExtractor import MedicalFeatutreExtractor
import torch
from LinearModel import LinearModel
import os
import logging
import jieba
import pandas as pd

logger = logging.getLogger(__name__)


class EvaluatorForTriple():
    def __init__(self, test_data_path: str, ent_path: str, device, has_label,
                 topks=[1, 10, 50, 100, 150, 200, 500, 1000, 1500, 2000]):
        self.device = device
        self.topks = topks
        self.has_label = has_label
        with open(ent_path, "r", encoding="utf8") as fr:
            self.ents = json.load(fr)
        print("transform to ent_set...")
        self.ent_set = None
        self.ent2sets()
        print("finish transform to ent_set")

        self.test_data = []
        line_ss_count = 2 if self.has_label else 1
        with open(test_data_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                if len(ss) == line_ss_count:
                    self.test_data.append(ss)
        # get set
        for i in self.test_data:
            sen = i[0]
            sen_seg = jieba.cut(sen)
            i.append(set(sen))  # -2
            i.append(set(sen_seg))  # -1
        # print(i)

    def ent2sets(self):
        """ 把实体库转换成纯集合类型 以方便快速计算 """
        self.ent_set = {}
        for k, ent_info in self.ents.items():
            v = {"name": None, "name_seg": None,
                 "company": None, "company_seg": None,
                 "bases": None, "bases_seg": None,
                 "functions": None, "functions_seg": None}
            name = ent_info["name"]
            if not (name == "None" or name == "nan" or len(name) < 1):
                v["name"] = set(name)
                v["name_seg"] = set(ent_info["name_seg"].split(" "))
            company = ent_info["company"]
            if not (company == "None" or company == "nan" or len(company) < 1):
                v["company"] = set(company)
                v["company_seg"] = set(ent_info["company_seg"].split(" "))
            bases = ent_info["bases"]
            if not (bases == "None" or bases == "nan" or len(bases) < 1):
                v["bases"] = set(bases)
                v["bases_seg"] = set(ent_info["bases_seg"].split(" "))
            functions = ent_info["functions"]
            if not (functions == "None" or functions == "nan" or len(functions) < 1):
                v["functions"] = set(functions)
                v["functions_seg"] = set(ent_info["functions_seg"].split(" "))
            self.ent_set[k] = v

    def eval(self, model, pred_save_path=None):
        model.eval()
        xls_data = []
        num_corrects = [0] * len(self.topks)
        with torch.no_grad():
            for c, i in enumerate(self.test_data):
                t = []
                t.extend(i[0:2])  # text label
                fes, subj_ids = [], []
                if c % 100 == 0:
                    print("evaluating...", c)
                for subj_id, value in self.ent_set.items():
                    subj_ids.append(str(subj_id).strip())
                    fes.append(MedicalFeatutreExtractor.get_feature_set(i[-2], i[-1], value))
                fes = torch.FloatTensor(fes).to(self.device)
                scores = model(fes).to("cpu").data.numpy()
                scores = [(float(scores[idx, 0]), subj_ids[idx]) for idx in range(len(subj_ids))]
                scores.sort(key=lambda x: x[0], reverse=True)
                sorted_subj_id = [x[1] for x in scores]
                t.append(",".join(sorted_subj_id[0:20]))
                t.append(i[1].strip() in sorted_subj_id[0:1000])
                for idx in range(len(self.topks)):
                    if i[1].strip() in sorted_subj_id[0:self.topks[idx]]:
                        num_corrects[idx] += 1
                xls_data.append(t)
        # 输出结果
        for idx in range(len(self.topks)):
            print(self.topks[idx], num_corrects[idx] / len(self.test_data))
        model.train()
        # 保存文件到本地
        if pred_save_path:
            pd.DataFrame(xls_data, columns=["Title", "Label", "Topk", "InTop1000"]).to_excel(pred_save_path,
                                                                                             index=False)

        return num_corrects[5] / len(self.test_data)

    def eval_test(self, model, save_path):
        model.eval()
        pred = []
        with torch.no_grad():
            for c, i in enumerate(self.test_data):
                fes, subj_ids = [], []
                if c % 100 == 0:
                    print("evaluating...", c)
                for subj_id, value in self.ent_set.items():
                    subj_ids.append(subj_id)
                    fes.append(MedicalFeatutreExtractor.get_feature_set(i[-2], i[-1], value))
                fes = torch.FloatTensor(fes).to(self.device)
                scores = model(fes).to("cpu").data.numpy()
                scores = [(float(scores[idx, 0]), subj_ids[idx]) for idx in range(len(subj_ids))]
                scores.sort(key=lambda x: x[0], reverse=True)
                sorted_subj_id = [x[1] for x in scores]
                pred.append(str(sorted_subj_id[0]).strip() + "\n")
        # 输出结果
        with open(save_path, "w", encoding="utf8") as fw:
            fw.writelines(pred)


if __name__ == "__main__":
    test_data_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\retrieval_interaction_manual_feature_data\dev.txt"
    kb_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_entity_kb.json"
    model_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\retrieval_interaction_manual_feature\mannual_feature\latest_model\fc_weight.bin"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel(12,model_path).to(device)
    evaluator = EvaluatorForTriple(test_data_path, kb_path, device, True)
    evaluator.eval(model, None)
