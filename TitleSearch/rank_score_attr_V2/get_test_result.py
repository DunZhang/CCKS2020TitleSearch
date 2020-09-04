import os
import torch
from transformers import BertTokenizer
from MixedEntScorerModel import MixedEntScorerModel
from TopKSearcherBasedRFIJSRes import TopKSearcherBasedRFIJSRes
from RSAV2Evaluator import RSAV2Evaluator
from BasesFeatureExtractor import BasesFeatureExtractor
from CompanyFeatureExtractor import CompanyFeatureExtractor
from CountryFeatureExtractor import CountryFeatureExtractor
from SpecFeatureExtractor import SpecFeatureExtractor

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    areas_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\areas.txt"
    file_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\test_data.txt"
    ent_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents_added_fun.bin"

    model_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\rank_score_attr_V2\rbt3_added_fun_no_share\best_model"
    tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, "name_attr_scoer"))
    model = MixedEntScorerModel(model_path, share_bert_weight=False).to(device)

    topk_searcher = TopKSearcherBasedRFIJSRes(
        r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\test_topk_ent_ids.bin")

    afes = {"company": CompanyFeatureExtractor(),
            "bases": BasesFeatureExtractor(),
            "place": CountryFeatureExtractor(areas_path),
            "spec": SpecFeatureExtractor()}
    evaluator = RSAV2Evaluator(file_path, ent_path, tokenizer, afes=afes, topk_searcher=topk_searcher, device=device,
                               task_name="test")
    evaluator.eval(model, xls_path=None, test_data_save_path="rbt3_fun_pred_no_share.txt")
