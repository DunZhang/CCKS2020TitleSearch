"""

训练分类器

"""
import logging

logging.basicConfig(level=logging.INFO)
import argparse
from Test import test

# if "win" in sys.platform:
#     PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
# else:
#     PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_epochs", default=5, type=int)
parser.add_argument("--num_labels", default=2, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--seq_length", default=50, type=int)
parser.add_argument("--warmup", default=0.1, type=float)

parser.add_argument("--evaluation_steps", default=100, type=int)  # 每多少步评估一次
parser.add_argument("--print_steps", default=20, type=int)  # 每多少步输出一次

parser.add_argument("--pretrain_model_path", default="/home/data_ti4_c/wangbin/bert_models/chinese_rbt3_pytorch/",
                    type=str)
parser.add_argument("--test_pretrain_model_path",
                    default="/home/data_ti4_c/wangbin/CCKS_dun_wb/TitleSearch/TitleTypeClassify/result/trian01/best_model/",
                    type=str)  # 预训练模型位置, 再尝试用超小模型train一下

parser.add_argument("--out_dir", default="./result/trian01", type=str)  # 保存模型位置
parser.add_argument("--train_file_path", default=r"./data/train.txt", type=str)
parser.add_argument("--dev_file_path", default=r"./data/dev.txt", type=str)
parser.add_argument("--test_file_path", default='./data/test.txt', type=str)

parser.add_argument("--device", default="6", type=str)

if __name__ == "__main__":
    conf = parser.parse_args()
    # args_dict = conf.__dict__
    # os.makedirs(conf.out_dir, exist_ok=True)
    # with open(os.path.join(conf.out_dir, 'setting_parameters.json'), 'w', encoding='utf-8')as f:
    #     f.write(json.dumps(args_dict, ensure_ascii=False, indent=4))
    # train(conf)
    test(conf)
