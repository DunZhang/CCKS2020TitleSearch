import os
import torch
from RankModel import RetrievalModel
from Evaluator import TopKAccEvaluator
from TopKSearcher import RFIJSTopKSearcher

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search_model_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\full_interaction_join_string\bert_min\best_model"
    search_fc_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\full_interaction_join_string\bert_min\best_model\fc_weight.bin"
    model_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\full_interaction_join_string\bert_base\best_model"
    fc_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\full_interaction_join_string\bert_base\best_model\fc_weight.bin"

    search_model = RetrievalModel(search_model_path, search_fc_path).to(device)
    model = RetrievalModel(model_path, fc_path).to(device)

    test_data_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\test_data.txt"
    ent_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_entity_kb.json"
    topk_searcher = RFIJSTopKSearcher(search_model, ent_path, search_model.tokenizer, device, 100, batch_size=2000, topk=200)

    eva = TopKAccEvaluator(test_data_path, ent_path, model.tokenizer, device, 100, 201, topk_searcher)
    eva.eval_test(model, "pred_test_base_use_topk.txt", "pred_test_base_use_topk.xlsx")
