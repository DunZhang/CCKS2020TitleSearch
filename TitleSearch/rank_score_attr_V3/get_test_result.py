import os
from os.path import join
import sys

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"
sys.path.append(join(PROJ_PATH, "TitleSearch"))
sys.path.append(join(PROJ_PATH, "TitleSearch/rank_score_attr_V2"))
sys.path.append(join(PROJ_PATH, "TitleSearch/rank_score_attr_V2/FeatureExtractors"))
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

    areas_path = join(PROJ_PATH, "data/external_data/areas.txt")
    file_path = join(PROJ_PATH, "data/format_data/tmp.txt")
    ent_path = join(PROJ_PATH, "data/format_data/medical_ents.bin")

    model_path = join(PROJ_PATH, "TitleSearch/rank_score_attr_V2/rbt3_added_fun/best_model")
    tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, "name_attr_scoer"))
    model = MixedEntScorerModel(model_path, share_bert_weight=False, strategy=None).to(device)

    topk_searcher = TopKSearcherBasedRFIJSRes(join(PROJ_PATH, "data/format_data/final_test_topk_ent_ids.bin"))

    afes = {"company": CompanyFeatureExtractor(),
            "bases": BasesFeatureExtractor(join(PROJ_PATH, "data/external_data/stop_words.txt")),
            "place": CountryFeatureExtractor(areas_path),
            "spec": SpecFeatureExtractor()}
    model.num_attr_fea = {k: v.get_num_features() for k, v in afes.items()}
    evaluator = RSAV2Evaluator(file_path, ent_path, tokenizer, afes=afes, topk_searcher=topk_searcher, device=device,
                               task_name="test", max_num_sen=None)
    evaluator.eval(model, xls_path="tmp.xlsx", test_data_save_path="tmp.txt")
