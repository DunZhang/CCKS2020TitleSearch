"""

训练分类器

"""
import logging

logging.basicConfig(level=logging.INFO)
import argparse
from train import train

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_epochs", default=3, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--margin", default=3.0, type=float)

parser.add_argument("--evaluation_steps", default=99999, type=int)  # 每多少步评估一次
parser.add_argument("--print_steps", default=100, type=int)  # 每多少步输出一次

parser.add_argument("--out_dir", default="mannual_feature", type=str)  # 保存模型位置
parser.add_argument("--train_file_path",
                    default=r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\retrieval_interaction_manual_feature_data\trian.txt",
                    type=str)
parser.add_argument("--kb_path",
                    default=r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_entity_kb.json",
                    type=str)
parser.add_argument("--dev_file_path",
                    default=r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\retrieval_interaction_manual_feature_data\dev.txt",
                    type=str)

parser.add_argument("--device", default="0", type=str)

if __name__ == "__main__":
    conf = parser.parse_args()
    train(conf)
    # train()
