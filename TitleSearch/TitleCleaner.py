import re
from openccpy.opencc import Opencc
import json
import pickle
import random


class TitleCleaner():
    def __init__(self):
        pass

    @classmethod
    def clean_title(cls, text: str) -> str:
        text = re.sub("\s", "", text)
        text = re.sub("&[0-9a-zA-Z]+?;", "", text)
        text = Opencc.to_simple(text)
        if len(text) < 1:
            return None
        return text

    @classmethod
    def clean_test_data(cls, read_path, save_path):
        """
        清理需要上传的测试数据
        :param read_path:
        :param save_path:
        :return:
        """
        data = []
        with open(read_path, "r", encoding="utf8") as fr:
            for line in fr:
                text = cls.clean_title(line)
                if text is None:
                    text = "空"
                data.append(text + "\n")
        with open(save_path, "w", encoding="utf8") as fw:
            fw.writelines(data)

    @classmethod
    def format_ori_data(cls, read_path, kb_path, train_save_path, dev_save_path, label_data_decoded_path=None):
        # get kb
        with open(kb_path, "rb") as fr:
            kb = pickle.load(fr)
        data = []
        ori_data = []
        text_id_set = set()
        with open(read_path, "r", encoding="utf8") as fr:
            for line in fr:
                di = json.loads(line)
                ori_data.append(str(di) + "\n")
                text = cls.clean_title(str(di["text"]))
                text_id = str(di["text_id"]).strip()
                if not text:
                    continue

                if len(di["implicit_entity"]) > 1:
                    print("有title对应多个ent")
                if text_id in text_id_set:
                    print("重复的text id", text_id)
                text_id_set.add(text_id)
                for item in di["implicit_entity"]:
                    label_id = str(item['subject_id']).strip()
                    if label_id in kb:
                        data.append("{}\t{}\n".format(text, label_id))
        for i in data:
            assert len(i.split("\t")) == 2
        random.shuffle(data)
        print(len(text_id_set))
        # with open(train_save_path, "w", encoding="utf8") as fw:
        #     fw.writelines(data[1000:])
        # with open(dev_save_path, "w", encoding="utf8") as fw:
        #     fw.writelines(data[:1000])
        #
        # if label_data_decoded_path:
        #     with open(label_data_decoded_path, "w", encoding="utf8") as fw:
        #         fw.writelines(ori_data)


if __name__ == "__main__":
    read_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\ori_data\train.txt"
    kb_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents.bin"
    train_save_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\train.txt"
    dev_save_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\dev.txt"
    # label_data_decoded_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_label_data_decode.txt"
    TitleCleaner.format_ori_data(read_path, kb_path, train_save_path, dev_save_path)
    # TitleCleaner.clean_test_data(r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\ori_data\dev.txt",
    #                              r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\test_data.txt")
