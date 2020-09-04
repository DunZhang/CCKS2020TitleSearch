"""

训练分类器

"""
import logging

logging.basicConfig(level=logging.INFO)
import argparse
from rsav2_train import train

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_epochs", default=3, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--margin", default=2.0, type=float)
parser.add_argument("--warmup", default=0.1, type=float)
parser.add_argument("--share_bert_weight", default=False, type=bool)

parser.add_argument("--evaluation_steps", default=3000, type=int)  # 每多少步评估一次
parser.add_argument("--print_steps", default=10, type=int)  # 每多少步输出一次

# parser.add_argument("--pretrain_model_path", default=r"G:\Data\roberta_tiny_clue", type=str)
parser.add_argument("--pretrain_model_path", default=r"G:\Data\RBT3", type=str)
# parser.add_argument("--pretrain_model_path", default=r"G:\Data\RoBERTa\torch\RoBERTa_zh_L12_PyTorch", type=str)
parser.add_argument("--train_topk_ent_path",
                    default=r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\train_topk_ent_ids.bin",
                    type=str)
parser.add_argument("--dev_topk_ent_path",
                    default=r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\dev_topk_ent_ids.bin",
                    type=str)

parser.add_argument("--out_dir", default="rbt3_added_fun_no_share", type=str)  # 保存模型位置
parser.add_argument("--train_file_path",
                    default=r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\train.txt",
                    type=str)
parser.add_argument("--ent_path",
                    default=r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents.bin",
                    type=str)
parser.add_argument("--dev_file_path",
                    default=r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\dev.txt",
                    type=str)
parser.add_argument("--test_file_path", default="", type=str)

parser.add_argument("--device", default="0", type=str)

if __name__ == "__main__":
    conf = parser.parse_args()
    train(conf)
    # train()
