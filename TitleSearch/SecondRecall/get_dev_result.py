import logging
from os.path import join

from TopKSearcherBasedRFIJSRes import TopKSearcher
from transformers import BertTokenizer
from RankEvaluator import SREvaluator
import os
import torch
from FirstScorerModel import FirstScorerModel

logging.basicConfig(level=logging.INFO)
import sys

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"

if __name__ == "__main__":
    file_path = join(PROJ_PATH, "data/rank_score_attr/dev.txt")
    ent_path = join(PROJ_PATH, "data/format_data/medical_ents.bin")
    gid2entids_path = join(PROJ_PATH, "data/format_data/g_id2ent_id.bin")
    entid2gid_path = join(PROJ_PATH, "data/format_data/ent_id2g_id.bin")
    topk_g_ids_path = join(PROJ_PATH, "data/format_data/topk_ent_ids.bin")
    topk_searcher = TopKSearcher(topk_g_ids_path, entid2gid_path)
    model_dir = join(PROJ_PATH, "TitleSearch/FirstScorer/rbt3_only_name/best_model/")

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FirstScorerModel(model_dir, join(model_dir, "fc_weight.bin")).to(device)
    model.eval()
    eva = SREvaluator(file_path=file_path, ent_path=ent_path, tokenizer=tokenizer,
                      topk_searcher=topk_searcher, device=device,
                      topks=[1, 2, 2, 3, 4, 5, 10, 20, 30, 50, 100, 200, 400, 600],
                      task_name="dev", max_num_sen=None, used_attrs=["name"], attr2cls_id={},
                      attr2max_len={"name": 60},
                      entid2gid_path=entid2gid_path, gid2entids_path=gid2entids_path)

    eva.eval(model, None, None)
