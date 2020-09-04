import os
import torch
from LinearModel import LinearModel
from Evaluator import EvaluatorForTriple
import logging

logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    test_data_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\test_data.txt"
    kb_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_entity_kb.json"
    model_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\retrieval_interaction_manual_feature\mannual_feature\latest_model\fc_weight.bin"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel(12, model_path).to(device)
    evaluator = EvaluatorForTriple(test_data_path, kb_path, device)
    evaluator.eval_test(model, "pred.txt")
