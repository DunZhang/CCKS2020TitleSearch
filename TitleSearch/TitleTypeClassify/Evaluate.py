import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score


def eval(model, dev_data_loader, device):
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for batch in dev_data_loader:
            batch_data = [i.to(device) for i in batch]
            logits, _ = model(input_ids=batch_data[0], attention_mask=batch_data[1],
                              token_type_ids=batch_data[2])
            logits = logits.to("cpu").numpy()
            labels.append(batch_data[3].to("cpu").numpy().reshape(-1, 1))
            preds.append(np.argmax(logits, axis=1).reshape(-1, 1))

    labels = np.vstack(labels)
    preds = np.vstack(preds)
    res = classification_report(labels, preds)
    acc = accuracy_score(labels, preds)
    model.train()
    return res, acc


if __name__ == "__main__":
    logits = np.random.randn(64, 10)
    labels = np.random.randint(0, 10, (64, 1))
    pred = np.argmax(logits, axis=1)
    print(pred.reshape(-1, 1).shape)
    # acc = accuracy_score(labels, pred)
    # res = classification_report(labels, pred)
    # print(res,acc)
