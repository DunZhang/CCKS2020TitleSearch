"""

训练分类器

"""
import logging

logging.basicConfig(level=logging.INFO)
import argparse
from Train import train
import sys
import os
from os.path import join

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_epochs", default=3, type=int)
parser.add_argument("--num_labels", default=10, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--seq_length", default=50, type=int)
parser.add_argument("--warmup", default=0.1, type=float)

parser.add_argument("--evaluation_steps", default=10, type=int)  # 每多少步评估一次
parser.add_argument("--print_steps", default=5, type=int)  # 每多少步输出一次

parser.add_argument("--pretrain_model_path", default=join(PROJ_PATH, "PreTrainedModels/roberta_tiny_clue"),
                    type=str)  # 预训练模型位置, 再尝试用超小模型train一下

parser.add_argument("--out_dir", default="tmp_test", type=str)  # 保存模型位置
parser.add_argument("--train_file_path", default=r"train.txt", type=str)
parser.add_argument("--dev_file_path", default=r"dev.txt", type=str)
parser.add_argument("--test_file_path", default=None, type=str)

parser.add_argument("--device", default="0", type=str)

if __name__ == "__main__":
    conf = parser.parse_args()
    train(conf)
    # train()
