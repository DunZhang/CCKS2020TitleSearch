import os
import torch
import logging
import json
from Evaluator import EvaluatorForTriple
import torch.nn.functional as F
from TripletDataSet import TripletDataSet
from LinearModel import LinearModel


def train(conf):
    ### device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### directory
    os.makedirs(conf.out_dir, exist_ok=True)
    best_model_dir = os.path.join(conf.out_dir, "best_model")
    os.makedirs(best_model_dir)
    latest_model_dir = os.path.join(conf.out_dir, "latest_model")
    os.makedirs(latest_model_dir)
    ### global variables
    best_acc = -1
    loss_fw = open(os.path.join(conf.out_dir, "loss.txt"), "w", encoding="utf8")
    acc_fw = open(os.path.join(conf.out_dir, "acc.txt"), "w", encoding="utf8")

    ### train data
    logging.info("get train data loader...")
    train_data_iter = TripletDataSet(conf.train_file_path, conf.kb_path, conf.batch_size)
    ### get dev evaluator
    evaluator = EvaluatorForTriple(conf.dev_file_path, conf.kb_path, conf.device)

    ### loss
    loss_model = torch.nn.CrossEntropyLoss().to(conf.device)

    ### model
    logging.info("define model...")
    model = LinearModel(12).to(conf.device)
    model.train()
    ### optimizer

    optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr)

    ### train
    global_step = 0
    logging.info("start train")
    for epoch in range(conf.num_epochs):
        for step, batch in enumerate(train_data_iter):
            global_step += 1
            step += 1
            batch_data = [i.to(conf.device) for i in batch]
            if step < 2:
                print(batch_data[0].shape)
            pos_score = model(batch_data[0])
            neg_score = model(batch_data[1])
            # triplet loss
            loss = torch.nn.functional.relu(neg_score - pos_score + conf.margin)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % conf.print_steps == 0:
                print("loss:", loss.item())
        # evaluate
        # acc = evaluator.eval(model)
        model.save(latest_model_dir)

    acc_fw.close()
    loss_fw.close()
