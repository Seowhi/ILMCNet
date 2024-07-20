import torch
# from net.model_net import Bert_CRF
# from net.t5_net import T5_CRF
from net.net import Net
from Io.data_loader import create_batch_iter
from train.train import fit
import config.args as args
from util.porgress_util import ProgressBar


def start(seq,task_name):
    args.seq = seq
    if(task_name == "PSSP_3"):
        args.task_name = "pssp_3"
    else:
        args.task_name = "pssp_8"
    print("The current task is :", args.task_name)

    train_iter, num_train_steps = create_batch_iter("train", args.TRAIN_PATH)
    eval_iter = create_batch_iter("dev", args.VALID_PATH)

    epoch_size = num_train_steps * args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs
    # print(f'epoch_size = {epoch_size}')
    pbar = ProgressBar(epoch_size=epoch_size, batch_size=args.train_batch_size)
    # model = T5_CRF(args.num_tags)



    return fit(
        training_iter=train_iter,
        eval_iter=eval_iter,
        num_epoch=args.num_train_epochs,
        pbar=pbar,
        num_train_steps=num_train_steps,
        verbose=1)
