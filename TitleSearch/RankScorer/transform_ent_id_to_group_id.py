import logging
from os.path import join
import pickle
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
import sys

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"


def get_entid_to_gid(ent_id_path, ent_id2gid_save_path, gid2entids_path):
    with open(ent_id_path, "rb") as fr:
        ents = pickle.load(fr)
    print(len(ents))
    name2ent_id = defaultdict(list)

    for ent_id, ent_info in ents.items():
        name = ent_info["name"]
        # name = name.replace("（", "(").replace("）", ")")
        # name = re.sub("\(.+?\)", "", name)
        name2ent_id[name].append(ent_id)
    print(len(name2ent_id))
    ent_id2g_id, g_id2ent_id = {}, {}
    for idx, g in enumerate(name2ent_id.values()):
        g_id2ent_id[str(idx)] = g
        for ent_id in g:
            ent_id2g_id[str(ent_id)] = str(idx)

    with open(ent_id2gid_save_path, "wb") as fw:
        pickle.dump(ent_id2g_id, fw)

    with open(gid2entids_path, "wb") as fw:
        pickle.dump(g_id2ent_id, fw)
    return ent_id2g_id


# def trans_titles(read_path, save_path, entid2gid_paath):
#     """
#     transform train, dev, test data
#     :param read_path:
#     :param save_path:
#     :return:
#     """
#     with open(entid2gid_paath, "rb") as fr:
#         ent_id_to_group_id = pickle.load(fr)
#     write_data = []
#     with open(read_path, "r", encoding="utf8") as fr:
#         for line in fr:
#             ss = line.strip().split("\t")
#             sen, ent_id = ss
#             g_id = ent_id_to_group_id[ent_id]
#             write_data.append("{}\t{}\n".format(sen, g_id))
#     with open(save_path, "w", encoding="utf8") as fw:
#         fw.writelines(write_data)


def trans_topk_end_ids(read_path, save_path, entid2gid_path):
    """
    train form topk ent ids
    :param read_path:
    :param save_path:
    :return:
    """
    with open(entid2gid_path, "rb") as fr:
        ent_id_to_group_id = pickle.load(fr)
    with open(read_path, "rb") as fr:
        topk_end_ids = pickle.load(fr)

    c1, c2 = 0, 0
    for k, v in topk_end_ids.items():
        c1 += len(v)
        t, t_set = [], set()
        for i in v:
            i = ent_id_to_group_id[i]
            if i not in t_set:
                t.append(i)
                t_set.add(i)
        topk_end_ids[k] = t
        c2 += len(topk_end_ids[k])

    print(c1 / len(topk_end_ids), c2 / len(topk_end_ids))
    with open(save_path, "wb") as fw:
        pickle.dump(topk_end_ids, fw)


if __name__ == "__main__":
    get_entid_to_gid(join(PROJ_PATH, "data/format_data/medical_ents.bin"),
                     join(PROJ_PATH, "data/format_data/ent_id2g_id.bin"),
                     join(PROJ_PATH, "data/format_data/g_id2ent_id.bin"))

    ##################################################################################
    # trans_topk_end_ids(join(PROJ_PATH, "data/format_data/topk_ent_ids.bin"),
    #                    join(PROJ_PATH, "data/format_data/topk_g_ids.bin"),
    #                    join(PROJ_PATH, "data/format_data/ent_id2g_id.bin"))
