"""

训练分类器

"""
import logging

logging.basicConfig(level=logging.INFO)
import argparse
from train import train

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--warmup", default=0.1, type=float)

parser.add_argument("--evaluation_steps", default=99999, type=int)  # 每多少步评估一次
parser.add_argument("--print_steps", default=5, type=int)  # 每多少步输出一次

# parser.add_argument("--pretrain_uq_model_path", default=r"G:\Data\RBT3", type=str)
# parser.add_argument("--pretrain_ent_model_path", default=r"G:\Data\RBT3", type=str)

# parser.add_argument("--pretrain_uq_model_path", default=r"G:\Data\simbert_torch", type=str)
# parser.add_argument("--pretrain_ent_model_path", default=r"G:\Data\simbert_torch", type=str)

parser.add_argument("--pretrain_uq_model_path", default=r"G:\Data\RoBERTa\torch\RoBERTa_zh_L12_PyTorch", type=str)
parser.add_argument("--pretrain_ent_model_path", default=r"G:\Data\RoBERTa\torch\RoBERTa_zh_L12_PyTorch", type=str)

parser.add_argument("--out_dir", default="saved_retrieval_dssm_tiny", type=str)  # 保存模型位置
parser.add_argument("--train_file_path", default=r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\retrieval\train.txt",
                    type=str)
parser.add_argument("--kb_path", default=r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\entity_kb.json",
                    type=str)
parser.add_argument("--dev_file_path", default=r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\retrieval\dev.txt",
                    type=str)
parser.add_argument("--test_file_path", default="", type=str)

parser.add_argument("--device", default="0", type=str)

if __name__ == "__main__":
    conf = parser.parse_args()
    train(conf)
    # train()
