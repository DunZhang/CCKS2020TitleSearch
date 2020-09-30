"""
训练分类器
"""
import logging
from os.path import join

logging.basicConfig(level=logging.INFO)
import sys

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"

sys.path.append(join(PROJ_PATH, "TitleSearch"))
sys.path.append(join(PROJ_PATH, "TitleSearch/FirstScorer"))
import argparse
from ListwiseTrain import listwise_train

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--num_epochs", default=6, type=int)
parser.add_argument("--attr2max_len", default=str({"name": 70}), type=str)
parser.add_argument("--used_attrs", default=str(["name"]), type=str)
parser.add_argument("--attr2cls_id", default=str({"name1": 1}), type=str)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--warmup", default=0.1, type=float)
parser.add_argument("--topk", default=10, type=int)
parser.add_argument("--num_neg_examples_per_record", default=9, type=int)
parser.add_argument("--shuffle", default=True, type=bool)

parser.add_argument("--evaluation_steps", default=3000, type=int)  # 每多少步评估一次
parser.add_argument("--print_steps", default=10, type=int)  # 每多少步输出一次

# parser.add_argument("--pretrain_model_path", default=join(PROJ_PATH, "PreTrainedModels/RoBERTa_zh_L12_PyTorch"),
#                     type=str)
parser.add_argument("--pretrain_model_path",
                    default=join(PROJ_PATH, "TitleSearch/FirstScorer/rbt_only_name_V1/best_model/"),
                    type=str)
# parser.add_argument("--pretrain_model_path", default=join(PROJ_PATH, "PreTrainedModels/RBT3"),
#                     type=str)
parser.add_argument("--train_topk_ent_path",
                    default=join(PROJ_PATH, "data/format_data/topk_ent_ids.bin"),
                    type=str)
parser.add_argument("--dev_topk_ent_path",
                    default=join(PROJ_PATH, "data/format_data/topk_ent_ids.bin"),
                    type=str)
parser.add_argument("--gid2entids_path",
                    default=join(PROJ_PATH, "data/format_data/g_id2ent_id.bin"),
                    type=str)
parser.add_argument("--entid2gid_path",
                    default=join(PROJ_PATH, "data/format_data/ent_id2g_id.bin"),
                    type=str)
# 保存模型位置
parser.add_argument("--out_dir", default="rbt_name_company_place_listwise_V1", type=str)
parser.add_argument("--max_num_sens", default=None, type=int)
parser.add_argument("--train_file_path",
                    default=join(PROJ_PATH, "data/format_data/train.txt"),
                    type=str)
parser.add_argument("--dev_file_path",
                    default=join(PROJ_PATH, "data/format_data/dev.txt"),
                    type=str)
parser.add_argument("--test_file_path", default="", type=str)

parser.add_argument("--ent_path",
                    default=join(PROJ_PATH, "data/format_data/medical_ents.bin"),
                    type=str)
parser.add_argument("--desc", default="",
                    type=str)
parser.add_argument("--device", default="0", type=str)

if __name__ == "__main__":
    conf = parser.parse_args()
    listwise_train(conf)
    # train()
    # print(conf.__dict__)
