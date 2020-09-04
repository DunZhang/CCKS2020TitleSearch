from transformers import BertConfig, BertModel
import torch
import logging

logging.basicConfig(level=logging.INFO)
conf = BertConfig(vocab_size=8021, hidden_size=312, num_hidden_layers=4,num_attention_heads=4,intermediate_size=256, )

model = BertModel(config=conf)
bert_tiny_weight = torch.load(r"G:\Data\roberta_tiny_clue\pytorch_model.bin")
di = model.state_dict()
# print(bert_tiny_weight.keys())
# print("embeddings.word_embeddings.weight" in di )
# print("embeddings.word_embeddings.weight" in bert_tiny_weight )
# di["embeddings.word_embeddings.weight"] = bert_tiny_weight["bert.embeddings.word_embeddings.weight"]
# di["embeddings.position_embeddings.weight"] = bert_tiny_weight["bert.embeddings.position_embeddings.weight"]
# di["embeddings.token_type_embeddings.weight"] = bert_tiny_weight["bert.embeddings.token_type_embeddings.weight"]
# di["embeddings.LayerNorm.weight"] = bert_tiny_weight["bert.embeddings.LayerNorm.weight"]
# di["embeddings.LayerNorm.bias"] = bert_tiny_weight["bert.embeddings.LayerNorm.bias"]
model.load_state_dict(di)
model.save_pretrained(".")