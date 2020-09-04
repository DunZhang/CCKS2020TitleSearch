import os
import torch
from transformers import BertTokenizer
from EntScorerModel import EntScorerModel
from TopKSearcherBasedRFIJSRes import TopKSearcherBasedRFIJSRes
from RSAEvaluator import RSAEvaluator

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\test_data.txt"
    ent_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents.bin"
    # model_path = r"G:\Data\roberta_tiny_clue"
    model_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\rank_score_attr\rbt3_relu_share_weight\best_model"
    tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, "bases_attr_scoer"))
    model = EntScorerModel(model_path, share_weight=True).to(device)

    topk_searcher = TopKSearcherBasedRFIJSRes(
        r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\test_topk_ent_ids.bin")
    # file_path, ent_path, model: EntScorerModel, tokenizer: BertTokenizer,
    # topk_searcher: ITopKSearcher, topks: List[int] = [1, 10, 20, 30, 50, 100, 200, 400, 600],
    # task_name: str = Union["dev", "test"])
    evaluator = RSAEvaluator(file_path, ent_path, tokenizer, topk_searcher=topk_searcher, device=device,
                             task_name="test")
    evaluator.eval(model, xls_path=None, test_data_save_path="pred_share_weight_70.txt")
