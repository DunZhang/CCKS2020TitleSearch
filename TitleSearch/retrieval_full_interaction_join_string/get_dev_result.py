import os
import torch
from RetrievalModel import RetrievalModel
from Evaluator import TopKAccEvaluator
from TopKSearcher import RFIJSTopKSearcher

if __name__ == "__main__":
    """
    1 0.54
10 0.852
20 0.9
30 0.928
50 0.944
100 0.964
200 0.984
400 0.988
600 0.988
1000 0.996
2000 1.0
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search_model_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\retrieval_full_interaction_join_string\bert_tiny\best_model"
    search_fc_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\retrieval_full_interaction_join_string\bert_tiny\best_model\fc_weight.bin"
    # model_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\full_interaction_join_string\bert_base\best_model"
    # fc_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\full_interaction_join_string\bert_base\best_model\fc_weight.bin"

    search_model = RetrievalModel(search_model_path, search_fc_path).to(device)
    # model = RetrievalModel(model_path, fc_path).to(device)

    test_data_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\retrieval_full_interaction_join_string\dev.txt"
    ent_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents.bin"
    # topk_searcher = TopKSearcher(search_model, ent_path, search_model.tokenizer, device, 100, batch_size=1000, topk=200)

    # eva = TopKAccEvaluator(test_data_path, ent_path, model.tokenizer, device, 100, 201, topk_searcher)
    eva = TopKAccEvaluator(test_data_path, ent_path, search_model.tokenizer, device, max_len=100, batch_size=100, )
    eva.eval_acc(search_model, "best_result.xlsx")
