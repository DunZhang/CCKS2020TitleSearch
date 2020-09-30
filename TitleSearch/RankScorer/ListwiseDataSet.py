import logging
from os.path import join
import pickle
from collections import defaultdict
import re
from ITopKSearcher import ITopKSearcher
import random
from TopKSearcher import TopKSearcher
from transformers import BertTokenizer
from typing import Iterable, List, Tuple, Dict
import torch
from multiprocessing import Pool
from DataUtil.DataUtil import DataUtil

logging.basicConfig(level=logging.INFO)
import sys

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"


class ListwiseDataSet():
    def __init__(self, file_path: str,
                 topk_searcher: ITopKSearcher, topk: int, num_neg_examples_per_record: int,
                 ent_path: str, gid2entids_path: str, entid2gid_path: str,
                 tokenizer: BertTokenizer, batch_size: int, attr2max_len: Dict, shuffle=True, used_attrs=["name"],
                 attr2cls_id=None):
        self.topk_searcher = topk_searcher
        self.topk = topk
        self.num_neg_examples_per_record = num_neg_examples_per_record
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.attr2max_len = attr2max_len
        self.shuffle = shuffle
        self.used_attrs = used_attrs
        self.use_multi_proc = True
        self.attr2cls_id = attr2cls_id
        self.pool = Pool(1)
        with open(ent_path, "rb") as fr:
            self.ents = pickle.load(fr)
        with open(gid2entids_path, "rb") as fr:
            self.gid2entids = pickle.load(fr)
        with open(entid2gid_path, "rb") as fr:
            self.entid2gid = pickle.load(fr)
        self.titles_info = []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                if len(ss) == 2:
                    self.titles_info.append(ss)
        self.examples = self.get_all_examples()
        if self.shuffle:
            random.shuffle(self.examples)
        self.steps = len(self.examples) // self.batch_size
        self.examples_iter = iter(self.examples)
        # 创建一个数据进程
        if self.use_multi_proc:
            batch_examples = self.get_batch_examples()
            self.proc = self.pool.apply_async(func=ListwiseDataSet.get_batch_data,
                                              args=(batch_examples, self.ents, self.tokenizer, self.attr2max_len,
                                                    self.used_attrs, self.attr2cls_id))

    # def __get_example_for_single_title(self, title, ent_id):
    #     """
    #     为单一title获取数据，朴素方法：随机采样，1正例，多负例
    #     :param title:
    #     :param ent_id:
    #     :return: List[ Tuple[title, pos_ent_id, List[neg_ent_ids]] ]
    #     """
    #     g_id = self.entid2gid[ent_id]
    #
    #     neg_ent_ids = []
    #     # get topk gids
    #     topk_gids = self.topk_searcher.get_topk_ids(title, self.num_neg_examples_per_record + 5)
    #     if g_id in topk_gids:
    #         topk_gids.remove(g_id)
    #     # get topk ent id
    #     for neg_g_id in topk_gids[0:self.num_neg_examples_per_record]:
    #         neg_ent_id = self.gid2entids[neg_g_id][-1]
    #         neg_ent_ids.append(neg_ent_id)  # 就选所有ents中的第0个的name，其实哪个都一样
    #     return title, ent_id, neg_ent_ids

    def __get_example_for_single_title(self, title, ent_id):
        """
        为单一title获取数据，朴素方法：随机采样，1正例，多负例
        :param title:
        :param ent_id:
        :return: List[ Tuple[title, pos_ent_id, List[neg_ent_ids]] ]
        """
        g_id = self.entid2gid[ent_id]

        neg_ent_ids = []
        # get topk gids
        topk_gids = self.topk_searcher.get_topk_g_ids(title, self.topk + 1)
        if g_id in topk_gids:
            topk_gids.remove(g_id)

        ################################################################################
        # 对topk_gids再做一次过滤，和目标title比较相似的直接过滤掉 用集合相似度
        # pos_ent_name_set = set(self.ents[ent_id]["name"])
        # t = []
        # for idx, neg_g_id in enumerate(topk_gids):
        #     if idx > 3:
        #         t.append(neg_g_id)
        #         continue
        #     neg_ent_name_set = set(self.ents[self.gid2entids[neg_g_id][-1]]["name"])
        #     sim = len(pos_ent_name_set.intersection(neg_ent_name_set)) / len(pos_ent_name_set.union(neg_ent_name_set))
        #     if sim < 0.8:
        #         t.append(neg_g_id)
        #     else:
        #         pass
        #         # print("remove", self.ents[pos_ent_id]["name"], self.ents[self.gid2entids[neg_g_id][-1]]["name"])
        # topk_gids = t
        ################################################################################
        # get topk ent id
        # for neg_g_id in random.sample(topk_gids, self.num_neg_examples_per_record):
        for neg_g_id in topk_gids[0:self.num_neg_examples_per_record]:
            neg_ent_id = self.gid2entids[neg_g_id][-1]
            neg_ent_ids.append(neg_ent_id)  # 就选所有ents中的第0个的name，其实哪个都一样
        while len(neg_ent_ids) != self.num_neg_examples_per_record:
            neg_ent_ids.append(neg_ent_ids[-1])
        return title, ent_id, neg_ent_ids

    def get_all_examples(self) -> List[Tuple[str, str, str]]:
        """
        为所有的title获取正实体和负实体,数据形如：
        [
        [title,pos_ent_id,neg_ent_id],
        [title,pos_ent_id,neg_ent_id1],
        [title,pos_ent_id,neg_ent_id2],
        [title,pos_ent_id,neg_ent_id3],
        [title1,pos_ent_id1,neg_ent_id4],
        ...
        []

        ]
        :return: List[ List[title, pos_ent_id, neg_ent_id] ]
        """
        random.shuffle(self.titles_info)
        examples = []
        for title, ent_id in self.titles_info:
            examples.append(self.__get_example_for_single_title(title, ent_id))
        return examples

    def get_batch_examples(self, ):
        """ 选取一批triple """
        examples = []
        for triple in self.examples_iter:
            examples.append(triple)
            if len(examples) >= self.batch_size:
                break
        if len(examples) == self.batch_size:
            return examples
        else:
            return []

    @staticmethod
    def get_batch_data(examples, ents, tokenizer: BertTokenizer, attr2max_len: Dict, used_attrs, attr2cls_id):
        """
        把一组triple 组合拼接为bert的输入
        :param examples:[ (title,posid,negids)  ]
        :param ents:
        :param tokenizer:
        :param attr2max_len:
        :param used_attrs:
        :return:pos_ipt, neg_ipt。
                {"name":{"input_ids":**, "attention_mask":**, "token_type_ids":**},
                {"function":{"input_ids":**, "attention_mask":**, "token_type_ids":**},

                }

        """
        if len(examples) < 1:
            return None
        ipt_list = []
        # 真实的batch是 batch_size * num_neg_example
        for title, pos_ent_id, neg_ent_ids in examples:
            example_ipts = []  # 一条example会产生多个ipt，第一个为正其他为负
            pos_ent = ents[pos_ent_id]
            example_ipts.append(
                ListwiseDataSet.trans_title_ent_to_bert_ipt(title, pos_ent, tokenizer, attr2max_len, used_attrs))
            for neg_ent_id in neg_ent_ids:
                neg_ent = ents[neg_ent_id]
                example_ipts.append(
                    ListwiseDataSet.trans_title_ent_to_bert_ipt(title, neg_ent, tokenizer, attr2max_len, used_attrs))
            ipt_list.extend(example_ipts)
        # concat 一下
        ipts = {attr_name: {} for attr_name in used_attrs}
        for attr_name in used_attrs:
            for ipt in ipt_list:
                for bert_ipt_name, value in ipt[attr_name].items():
                    if bert_ipt_name not in ipts[attr_name]:
                        ipts[attr_name][bert_ipt_name] = [value]
                    else:
                        ipts[attr_name][bert_ipt_name].append(value)
            # 组合成2维数组，batch_size*max_len
            for bert_ipt_name, value in ipts[attr_name].items():
                ipts[attr_name][bert_ipt_name] = torch.cat(value, dim=0)
            # 把cls变成专门的ID
            if attr2cls_id is not None and attr_name in attr2cls_id:
                ipts[attr_name]["input_ids"][:, 0] = attr2cls_id[attr_name]
        return ipts

    @staticmethod
    def trans_title_ent_to_bert_ipt(title: str, ent_info, tokenizer: BertTokenizer, attr2max_len: Dict,
                                    used_attrs: List[str]):
        """

        :param title:
        :param ent_info:
        :param tokenizer:
        :param attr2max_len:
        :param used_attrs:
        :return:Dict[attr_name:{"input_ids":..}]
        """
        ipt = {}
        for attr_name in used_attrs:
            attr_value = ent_info[attr_name]
            ##########################################################################################################
            if not DataUtil.is_null(ent_info["company"]):
                attr_value = ent_info["company"] + "。" + attr_value
            if not DataUtil.is_null(ent_info["place"]):
                attr_value = ent_info["place"] + "。" + attr_value
            #########################################################################################################
            ipt[attr_name] = tokenizer.encode_plus(title, attr_value, max_length=attr2max_len[attr_name],
                                                   padding="max_length", truncation=True, return_tensors="pt")
        return ipt

    def __iter__(self):
        return self

    def __next__(self):
        # 获取数据
        if self.use_multi_proc:
            ipt = self.proc.get()
            batch_examples = self.get_batch_examples()
            # print("len(batch_examples)",len(batch_examples))
            self.proc = self.pool.apply_async(func=ListwiseDataSet.get_batch_data,
                                              args=(batch_examples, self.ents, self.tokenizer, self.attr2max_len,
                                                    self.used_attrs, self.attr2cls_id))
        else:
            batch_examples = self.get_batch_examples()
            ipt = ListwiseDataSet.get_batch_data(batch_examples, self.ents, self.tokenizer, self.attr2max_len,
                                                 self.used_attrs, self.attr2cls_id)
        # 重置迭代器
        if not ipt:
            self.examples = self.get_all_examples()
            if self.shuffle:
                random.shuffle(self.examples)
            self.examples_iter = iter(self.examples)
            if self.use_multi_proc:
                self.proc.get()
                batch_examples = self.get_batch_examples()
                self.proc = self.pool.apply_async(func=ListwiseDataSet.get_batch_data,
                                                  args=(batch_examples, self.ents, self.tokenizer, self.attr2max_len,
                                                        self.used_attrs, self.attr2cls_id))
            raise StopIteration
        else:
            return ipt


if __name__ == "__main__":
    file_path = join(PROJ_PATH, "data/rank_score_attr/dev.txt")
    topk = 25
    num_neg_examples_per_record = 4
    ent_path = join(PROJ_PATH, "data/format_data/medical_ents.bin")
    gid2entids_path = join(PROJ_PATH, "data/format_data/g_id2ent_id.bin")
    entid2gid_path = join(PROJ_PATH, "data/format_data/ent_id2g_id.bin")
    topk_ent_ids_path = join(PROJ_PATH, "data/format_data/topk_ent_ids.bin")
    topk_searcher = TopKSearcher(topk_ent_ids_path, entid2gid_path)
    model_dir = join(PROJ_PATH, "PreTrainedModels/RBT3")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    batch_size = 6
    # attr2max_len = {"name": 80, "functions": 100}
    # used_attrs = ["name", "functions"]
    attr2max_len = {"name": 80}
    used_attrs = ["name"]
    t = ListwiseDataSet(file_path,
                        topk_searcher, topk, num_neg_examples_per_record,
                        ent_path, gid2entids_path, entid2gid_path,
                        tokenizer, batch_size, attr2max_len, shuffle=True, used_attrs=used_attrs,
                        attr2cls_id={"name": 1, "functions": 2})

    for step, ipt in enumerate(t):
        print("=====================================================================================")
        print(ipt["name"]["input_ids"].shape)
        print(tokenizer.decode(ipt["name"]["input_ids"][0].data.numpy().tolist()))
        print(tokenizer.decode(ipt["name"]["input_ids"][1].data.numpy().tolist()))
        print(tokenizer.decode(ipt["name"]["input_ids"][2].data.numpy().tolist()))
        print(tokenizer.decode(ipt["name"]["input_ids"][3].data.numpy().tolist()))
        print(tokenizer.decode(ipt["name"]["input_ids"][4].data.numpy().tolist()))
        print(tokenizer.decode(ipt["name"]["input_ids"][5].data.numpy().tolist()))
        print(ipt["name"]["input_ids"][0].data.numpy().tolist())
        print(ipt["name"]["attention_mask"][0])
        print(ipt["name"]["token_type_ids"][0])
        print("=====================================================================================")
        # print(ipt["functions"]["input_ids"].shape)
        # print(tokenizer.decode(ipt["functions"]["input_ids"][0].data.numpy().tolist()))
        # print(ipt["functions"]["input_ids"][0].data.numpy().tolist())
        # print(ipt["functions"]["attention_mask"][0])
        # print(ipt["functions"]["token_type_ids"][0])

        # x = input()
        # if "stop" in x:
        #     exit(0)
