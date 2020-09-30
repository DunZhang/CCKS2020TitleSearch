import os
import torch
import logging
from transformers import BertTokenizer
from SRScorerModel import SRScorerModel

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from SRTripleDataSet import SRTripleDataSet
from TopKSearcherBasedRFIJSRes import TopKSearcher
from SREvaluator import SREvaluator


def sr_triple_train(conf):
    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # directory
    os.makedirs(conf.out_dir, exist_ok=True)
    with open(os.path.join(conf.out_dir, "README.txt"), "w", encoding="utf8") as fw:
        for k, v in conf.__dict__.items():
            fw.write("{}:\t{}\n".format(k, v))

    best_model_dir = os.path.join(conf.out_dir, "best_model")
    os.makedirs(best_model_dir)
    latest_model_dir = os.path.join(conf.out_dir, "latest_model")
    os.makedirs(latest_model_dir)
    # global variables
    best_acc = -1
    loss_fw = open(os.path.join(conf.out_dir, "loss.txt"), "w", encoding="utf8")
    acc_fw = open(os.path.join(conf.out_dir, "acc.txt"), "w", encoding="utf8")

    # topk seacher
    logging.info("get topk searcher")
    train_topk_searcher = TopKSearcher(conf.train_topk_ent_path, conf.entid2gid_path)
    dev_topk_searcher = TopKSearcher(conf.dev_topk_ent_path, conf.entid2gid_path)

    # train data
    logging.info("get train data loader...")
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_model_path)
    train_data_iter = SRTripleDataSet(file_path=conf.train_file_path,
                                      topk_searcher=train_topk_searcher, topk=conf.topk,
                                      num_neg_examples_per_record=conf.num_neg_examples_per_record,
                                      ent_path=conf.ent_path, gid2entids_path=conf.gid2entids_path,
                                      entid2gid_path=conf.entid2gid_path, tokenizer=tokenizer, batch_size=conf.batch_size,
                                      attr2max_len=eval(conf.attr2max_len), shuffle=conf.shuffle,
                                      used_attrs=eval(conf.used_attrs), attr2cls_id=eval(conf.attr2cls_id))

    total_steps = int(train_data_iter.steps * conf.num_epochs)
    steps_per_epoch = train_data_iter.steps
    if conf.warmup < 1:
        warmup_steps = int(total_steps * conf.warmup)
    else:
        warmup_steps = int(conf.warmup)
    # get dev evaluator
    evaluator = SREvaluator(file_path=conf.dev_file_path, ent_path=conf.ent_path, tokenizer=tokenizer,
                            topk_searcher=dev_topk_searcher, device=conf.device, topk=conf.topk,
                            task_name="dev", entid2gid_path=conf.entid2gid_path,
                            gid2entids_path=conf.gid2entids_path, attr2max_len=eval(conf.attr2max_len),
                            used_attrs=eval(conf.used_attrs), attr2cls_id=eval(conf.attr2cls_id))
    # model
    logging.info("define model...")
    model = SRScorerModel(conf.pretrain_model_path).to(conf.device)
    model.train()
    # optimizer
    logging.info("define optimizer...")
    no_decay = ["bias", "LayerNorm.weight"]
    paras = dict(model.named_parameters())
    logging.info("=================================== trained parameters ===================================")
    for n, p in paras.items():
        logging.info("{}:\t{}".format(n, p.shape))
    optimizer_grouped_parameters = [{
        "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
        {"params": [p for n, p in paras.items() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # train
    global_step = 0
    logging.info("start train")
    for epoch in range(conf.num_epochs):
        for step, (pos_ipt, neg_ipt) in enumerate(train_data_iter):
            global_step += 1
            step += 1
            # 送入到device中
            for attr_name in eval(conf.used_attrs):
                for bert_ipt_name in pos_ipt[attr_name]:
                    pos_ipt[attr_name][bert_ipt_name] = pos_ipt[attr_name][bert_ipt_name].to(conf.device)
            for attr_name in eval(conf.used_attrs):
                for bert_ipt_name in neg_ipt[attr_name]:
                    neg_ipt[attr_name][bert_ipt_name] = neg_ipt[attr_name][bert_ipt_name].to(conf.device)
            pos_score = None
            for attr_name in eval(conf.used_attrs):
                if pos_score is not None:
                    pos_score = pos_score + model(**pos_ipt[attr_name])
                else:
                    pos_score = model(**pos_ipt[attr_name])
            neg_score = None
            for attr_name in eval(conf.used_attrs):
                if neg_score is not None:
                    neg_score = neg_score + model(**neg_ipt[attr_name])
                else:
                    neg_score = model(**neg_ipt[attr_name])
            if step < 2 or step % 800 == 0:
                print("=====================================================================================")
                print(pos_score.shape)
                print(pos_ipt["name"]["input_ids"].shape)
                print(tokenizer.decode(pos_ipt["name"]["input_ids"][0].cpu().data.numpy().tolist()))
                print(pos_ipt["name"]["input_ids"][0].cpu().data.numpy().tolist())
                print(pos_ipt["name"]["attention_mask"][0].cpu())
                print(pos_ipt["name"]["token_type_ids"][0].cpu())
                print("------------------------------------------------------------------------------------")
                print(tokenizer.decode(neg_ipt["name"]["input_ids"][0].cpu().data.numpy().tolist()))
                print(neg_ipt["name"]["input_ids"][0].cpu().data.numpy().tolist())
                print(neg_ipt["name"]["attention_mask"][0].cpu())
                print(neg_ipt["name"]["token_type_ids"][0].cpu())

                # print("=====================================================================================")
                # print(pos_ipt["functions"]["input_ids"].shape)
                # print(tokenizer.decode(pos_ipt["functions"]["input_ids"][0].cpu().data.numpy().tolist()))
                # print(pos_ipt["functions"]["input_ids"][0].cpu().data.numpy().tolist())
                # print(pos_ipt["functions"]["attention_mask"][0].cpu())
                # print(pos_ipt["functions"]["token_type_ids"][0].cpu())
                # print("------------------------------------------------------------------------------------")
                # print(tokenizer.decode(neg_ipt["functions"]["input_ids"][0].cpu().data.numpy().tolist()))
                # print(neg_ipt["functions"]["input_ids"][0].cpu().data.numpy().tolist())
                # print(neg_ipt["functions"]["attention_mask"][0].cpu())
                # print(neg_ipt["functions"]["token_type_ids"][0].cpu())

            loss = torch.nn.functional.relu(neg_score - pos_score + conf.margin)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([v for k, v in paras.items()], max_norm=1)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()
            if step % conf.print_steps == 0:
                logging.info("epoch:{},\tstep:{}/{},\tloss:{}".format(epoch, step, steps_per_epoch, loss.data))
                loss_fw.write("epoch:{},\tstep:{}/{},\tloss:{}\n".format(epoch, step, steps_per_epoch, loss.data))
                loss_fw.flush()

            if step % conf.evaluation_steps == 0:
                logging.info("start evaluate...")
                acc = evaluator.eval(model)
                if acc > best_acc:
                    logging.info("save best model to {}".format(best_model_dir))
                    best_acc = acc
                    model.save(best_model_dir)
                model.save(latest_model_dir)
                acc_fw.write("epoch:{},\tstep:{}/{},\tacc:{}\n".format(epoch, step, steps_per_epoch, acc))
                acc_fw.flush()

        logging.info("start evaluate...")
        acc = evaluator.eval(model)
        if acc > best_acc:
            logging.info("save best model to {}".format(best_model_dir))
            best_acc = acc
            model.save(best_model_dir)
        model.save(latest_model_dir)
        acc_fw.write("epoch:{},\tstep:{}/{},\tacc:{}\n".format(epoch, step, steps_per_epoch, acc))
        acc_fw.flush()
    acc_fw.close()
    loss_fw.close()
