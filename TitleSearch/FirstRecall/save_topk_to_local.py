""" 使用检索模型计算所有句子的topk """
import os
import torch
from RetrievalModel import RetrievalModel
from RFIJSTopKSearcher import RFIJSTopKSearcher
import pickle

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"G:\Codes\CCKS2020TitleSearch\TitleSearch\retrieval_full_interaction_join_string\bert_tiny\best_model"
    fc_path = r"G:\Codes\CCKS2020TitleSearch\TitleSearch\retrieval_full_interaction_join_string\bert_tiny\best_model\fc_weight.bin"
    sens_path = r"G:\Codes\CCKS2020TitleSearch\data\format_data\final_test.txt"
    save_path = r"G:\Codes\CCKS2020TitleSearch\data\format_data\final_test_topk_ent_ids.bin"
    search_model = RetrievalModel(model_path, fc_path).to(device)

    ent_path = r"G:\Codes\CCKS2020TitleSearch\data\format_data\medical_ents.bin"

    topk_searcher = RFIJSTopKSearcher(search_model, ent_path, search_model.tokenizer, device, max_len=100, batch_size=1000,
                                      topk=2000)

    # get all topk ent ids sens

    title2topk_ent_ids = {}
    c = 0
    with open(sens_path, "r", encoding="utf8") as fr:
        for idx, line in enumerate(fr):
            if "\t" in line:
                print("ERROR", line)
            # title, ent_id = line.strip().split("\t")
            title = line.strip().split("\t")[0]
            topk_ent_ids = topk_searcher.get_topk_ids(title)
            # c += 1 if ent_id in topk_ent_ids[0:50] else 0
            title2topk_ent_ids[title] = topk_ent_ids
            if idx % 20 == 0:
                print(idx, title)
    print(c / len(title2topk_ent_ids))
    with open(save_path, "wb") as fw:
        pickle.dump(title2topk_ent_ids, fw)
