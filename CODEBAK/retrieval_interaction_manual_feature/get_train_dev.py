import random


def get_train_dev(read_path, train_save_path, dev_save_path):
    data = []
    with open(read_path, "r", encoding="utf8") as fr:
        for line in fr:
            ss = line.strip().split("\t")
            if len(ss) == 2:
                data.append(line)
    random.shuffle(data)
    train_data, dev_data = data[250:], data[0:250]
    with open(train_save_path, "w", encoding="utf8") as fw:
        fw.writelines(train_data)
    with open(dev_save_path, "w", encoding="utf8") as fw:
        fw.writelines(dev_data)


if __name__ == "__main__":
    get_train_dev(r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_label_data.txt",
                  r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\retrieval_interaction_manual_feature_data\train.txt",
                  r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\retrieval_interaction_manual_feature_data\dev.txt")
