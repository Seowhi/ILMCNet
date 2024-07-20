import time
import warnings
import os

import numpy as np
import torch

from pytorch_pretrained_bert.optimization import BertAdam
import config.args as args
from net.net import Net
from transformers import T5Tokenizer, T5EncoderModel
from util.plot_util import loss_acc_plot
from util.Logginger import init_logger
from util.model_util import save_model
from Io.data_loader import single_to_features
from get_prottrans import get_single_embeddings

logger = init_logger("torch", logging_path=args.log_path)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
warnings.filterwarnings('ignore')

WEIGHTS_NAME = 'pytorch_model.bin'
CONFIG_NAME = 'bert_config.json'


BASES = 'ARNDCQEGHILKMFPSTWYVX'

def one_hot(sequence, max_length):

    def amino_acid_to_one_hot(amino_acid):
        if amino_acid not in BASES:

            return torch.tensor([0] * 21, dtype=torch.float)
        index = BASES.index(amino_acid)
        return torch.tensor([0] * 21, dtype=torch.float).index_fill_(0, torch.tensor([index]), 1)


    one_hot_list = [amino_acid_to_one_hot(aa) for aa in sequence]

    one_hot_encoded = torch.stack(one_hot_list)


    if one_hot_encoded.size(0) < max_length:
        padding_size = max_length - one_hot_encoded.size(0)
        one_hot_encoded = torch.cat((one_hot_encoded, torch.zeros(padding_size, 21, dtype=torch.float)), dim=0)

    return one_hot_encoded

def get_output(tensor):
    labels = args.labels

    mapped_labels = [labels[idx] for idx in tensor]

    output_string = ''.join(mapped_labels)
    return output_string

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def map_tensor_to_labels(tensor):

    label_map = {
        5: 'L', 6: 'A', 7: 'G', 8: 'V', 9: 'E', 10: 'S', 11: 'I', 12: 'K', 13: 'R', 14: 'D',
        15: 'T', 16: 'P', 17: 'N', 18: 'Q', 19: 'F', 20: 'Y', 21: 'M', 22: 'H', 23: 'C', 24: 'W',
        25: 'X', 26: 'U', 27: 'B', 28: 'Z', 29: 'O'
    }


    filtered_tensor = tensor[1:]
    filtered_tensor = filtered_tensor[(filtered_tensor != 0) & (filtered_tensor != 3)]


    labels = [label_map.get(item.item(), 'UNK') for item in filtered_tensor]

    return ''.join(labels)

def get_embeddings(input_ids, task_name):

    if args.embds == 'esm':
        str_name = '_esm'
        tensors = []
        directory = "/media/sda1/dongbenzhi/sh/t5lstm-crf/esm"

        for input_id in input_ids:
            pdb_id = str(input_id[0].item() - 30) + str_name
            # print("pdb_id shape:", pdb_id)


            for file in os.listdir(directory):
                if file.endswith(".npy") and os.path.splitext(file)[0] == pdb_id:
                    file_path = os.path.join(directory, file)
                    tensor = np.load(file_path, allow_pickle=True)
                    # print("tensor shape:", tensor.shape)  # torch.Size([880, 1280])

            tensor = torch.from_numpy(tensor)

            tensors.append(tensor)

        tensors_list = torch.stack(tensors, dim=0)
        # print(tensors_list)
        # print("tensors_list shape:", tensors_list.shape)
        # exit()

        return tensors_list
    else:


        if task_name == 'train':
            str_name = args.train_dataset + "|"
            data = np.load(args.npz_path, allow_pickle=True)
            max_length = args.max_seq_length
        else:
            str_name = args.test_dataset.upper() + "|"
            data = np.load(f'/media/sda1/dongbenzhi/sh/t5lstm-crf/t5/{args.test_dataset}.npz', allow_pickle=True)
            max_length = args.max_valid_length

        tensors = []

        for input_id in input_ids:
            if args.train_dataset == "cullpdb":
                index = 30
            else:
                index = 5461
            pdb_id = str_name + str(input_id[0].item() - index)
            tensor = data[pdb_id]   #torch.Size([880, 1024])

            # seq = map_tensor_to_labels(input_id.cpu())
            # tensor_onehot = one_hot(seq, max_length)  # torch.Size([880, 20])
            # tensor = torch.cat((torch.from_numpy(tensor), tensor_onehot), dim=-1) #torch.Size([880, 1044])
            #
            tensor = torch.from_numpy(tensor)
            tensors.append(tensor)

        tensors_list = torch.stack(tensors, dim=0)
        # print(tensors_list)
        # print("tensors_list shape:", tensors_list.shape)
        # exit()

        return tensors_list, max_length

def read_fasta_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if len(lines) < 2:
        raise ValueError("File does not contain enough lines to extract seq_name and seq.")

    header = lines[0].strip()
    sequence = lines[1].strip()

    if not header.startswith('>'):
        raise ValueError("Header line does not start with '>'.")

    seq_name = header.split('|')[0][1:]

    return seq_name, sequence

def parse_segments(sequence, types):
    segments = {t: [] for t in types}
    current_type = None
    start = None

    for i, s in enumerate(sequence):
        if s != current_type:
            if current_type in types and start is not None:
                segments[current_type].append((start, i - 1))
            start = i
            current_type = s

    if current_type in types:
        segments[current_type].append((start, len(sequence) - 1))

    return segments

def calculate_sov(true_tensor, pred_tensor, types):

    true_seq = true_tensor.tolist()
    pred_seq = pred_tensor.tolist()


    true_seq = ''.join(map(str, true_seq))
    pred_seq = ''.join(map(str, pred_seq))
    # print(true_seq)
    # print(pred_seq)
    # exit()
    true_segments = parse_segments(true_seq, types)
    pred_segments = parse_segments(pred_seq, types)
    total_true_length = 0
    total_score = 0

    for t in types:
        for true_segment in true_segments[t]:
            true_start, true_end = true_segment
            true_length = true_end - true_start + 1
            total_true_length += true_length
            max_score = 0

            for pred_segment in pred_segments[t]:
                pred_start, pred_end = pred_segment
                overlap_start = max(true_start, pred_start)
                overlap_end = min(true_end, pred_end)
                overlap_length = max(0, overlap_end - overlap_start + 1)

                if overlap_length > 0:
                    len_diff = abs((true_end - true_start) - (pred_end - pred_start))
                    delta = min(overlap_length, overlap_length + len_diff)
                    score = (overlap_length * delta) / true_length
                    max_score = max(max_score, score)

            total_score += max_score

    if total_true_length == 0:
        return 0  # Avoid division by zero
    return (total_score / total_true_length) * 100


def fit(training_iter, eval_iter, num_epoch, pbar, num_train_steps, verbose=1):
    Result = "null"
    model = Net()
    # model = Bert_CRF.from_pretrained(args.bert_model, num_tag=len(args.labels))
    print("pre_model_name:", args.model)
    print("\n")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    if args.is_train == False:
        model.load_state_dict(torch.load(args.model_path))

    # ------------------CUDA----------------------
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # ---------------------optimizer-------------------------
    # print(model)
    # print('------------------------')
    # print(dir(model.T5))

    # print(model.named_parameters())
    # param_optimizer = list(model.T5.named_parameters())
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #
    # optimizer_grouped_parameters = [{
    #     'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #     'weight_decay': 0.01
    # }, {
    #     'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #     'weight_decay': 0.0
    # }]
    #
    t_total = num_train_steps

    ## ---------------------GPUfp16-----------------------------
    # if args.fp16:
    #     try:
    #         from apex.optimizers import FP16_Optimizer
    #         from apex.optimizers import FusedAdam
    #     except ImportError:
    #         raise ImportError(
    #             "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
    #         )
    #
    #     optimizer = FusedAdam(optimizer_grouped_parameters,
    #                           lr=args.learning_rate,
    #                           bias_correction=False,
    #                           max_grad_norm=1.0)
    #     if args.loss_scale == 0:
    #         optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    #     else:
    #         optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    # ## ------------------------GPUfp32---------------------------
    # else:
    #     optimizer = BertAdam(optimizer_grouped_parameters,
    #                          lr=args.learning_rate,
    #                          warmup=args.warmup_proportion,
    #                          t_total=t_total)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    # ---------------------model----------------------
    if args.fp16:
        model.half()

    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # device = torch.device("cpu")
    n_gpu = torch.cuda.device_count()

    logger.info(f"Device {device} n_gpu {n_gpu} distributed training")

    model.to(device)
    # T5 = T5EncoderModel.from_pretrained(args.bert_model, local_files_only=True, torch_dtype=torch.float16)
    # T5 = T5.to(device).eval()
    # tokenizer = T5Tokenizer.from_pretrained(args.bert_model, local_files_only=True,
    #                                              do_lower_case=False)


    if n_gpu > 1:
        # model = torch.nn.DataParallel(model)
        # model_module = model.module
        model_module = model
    else:
        model_module = model

    train_losses = []
    eval_losses = []
    train_accuracy = []
    eval_accuracy = []

    history = {
        "train_loss": train_losses,
        "train_acc": train_accuracy,
        "eval_loss": eval_losses,
        "eval_acc": eval_accuracy
    }


    start = time.time()
    global_step = 0
    best_loss = 10
    train_acc = 0
    train_loss = 0


    for e in range(num_epoch):

    # ------------------------train------------------------------
        if args.is_train == True:
            print("Epoch:", e + 1)
            model.train()
            for step, batch in enumerate(training_iter):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, output_mask = batch

                tensors_pt, max_length = get_embeddings(input_ids,"train")     # torch.Size([2,880, 1024])
                tensors_seq = model_module.get_embeddings(input_ids, segment_ids, max_length)#torch.Size([2,880, 1024])
                tensors = torch.cat((tensors_pt, tensors_seq.cpu()), dim=-1)#torch.Size([2,880, 2048])


                encode = model(tensors) # torch.Size([2, 700, 3])
                # encode = model(tensors_pt)


                # bert_encode = model(input_ids, segment_ids, input_mask).cpu()
                # bert_encode = model(input_ids, segment_ids, input_mask).to(device)  # torch.Size([2, 700, 8])

                train_loss = model_module.loss_fn(encode=encode,
                                                  tags=label_ids,
                                                  output_mask=output_mask)
                if args.gradient_accumulation_steps > 1:
                    train_loss = train_loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(train_loss)
                else:
                    train_loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                predicts = model_module.predict(encode, output_mask)

                label_ids = label_ids.view(1, -1)
                label_ids = label_ids[label_ids != -1]
                label_ids = label_ids.cpu()
                # label_ids = torch.cat([label_ids, pad_predicts], dim=0)

                # if len(predicts) != len(label_ids):
                #     continue
                # print(len(predicts))
                # print(len(label_ids))
                # print(predicts)
                # print(label_ids)
                # exit()

                train_acc, f1 = model_module.acc_f1(predicts, label_ids)
                pbar.show_process(train_acc, train_loss.item(), f1, time.time() - start, step)

    # -----------------------valid----------------------------
        if args.is_valid == True:
            # print("Valid Dataset: ", args.test_dataset)
            model.eval()
            count = 0
            score = 0.75
            y_predicts, y_labels = [], []
            eval_loss, eval_acc, eval_f1 = 0, 0, 0
            sov_score = 0
            with torch.no_grad():
                for step, batch in enumerate(eval_iter):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids, output_mask = batch

                    tensors_pt, max_length = get_embeddings(input_ids, "valid")  # torch.Size([2,700, 1024])
                    tensors_seq = model_module.get_embeddings(input_ids, segment_ids, max_length) #torch.Size([2,880, 1024])
                    tensors = torch.cat((tensors_pt, tensors_seq.cpu()), dim=-1)  #torch.Size([2,880, 2048])

                    encode = model(tensors) # torch.Size([2, 700, 3])
                    # encode = model(tensors_pt)

                    eval_los = model_module.loss_fn(encode=encode,
                                                    tags=label_ids,
                                                    output_mask=output_mask)
                    eval_loss = eval_los + eval_loss
                    count += 1
                    predicts = model_module.predict(encode, output_mask)

                    label_ids = label_ids.view(1, -1)
                    label_ids = label_ids[label_ids != -1].cpu()
                    # if len(predicts) != len(label_ids):
                    #     continue
                    y_predicts.append(predicts)
                    y_labels.append(label_ids)
                    # print(len(predicts))
                    # print(len(label_ids))
                    # print(predicts)
                    # print(label_ids)
                    # exit()
                    train_acc, f1 = model_module.acc_f1(predicts, label_ids)
                    sov_score += calculate_sov(label_ids, predicts, args.types)
                    if args.num_tags == 8:
                        score = 0.6
                    if (train_acc < score):
                        for input_id in input_ids:
                            if args.train_dataset == "cullpdb":
                                index = 30
                            else:
                                index = 5461
                            pdb_id = str(input_id[0].item() - index)
                            print(f"{pdb_id}  {train_acc:.4f}  {predicts}  {label_ids}")


                eval_predicted = torch.cat(y_predicts, dim=0).cpu()
                eval_labeled = torch.cat(y_labels, dim=0).cpu()

                eval_acc, eval_f1 = model_module.acc_f1(eval_predicted, eval_labeled)
                model_module.class_report(eval_predicted, eval_labeled)

                # print('\nEpoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - eval_f1:%4f\n'
                #       %
                #       (e + 1, train_loss.item(), eval_loss.item() / count, train_acc, eval_acc, eval_f1))
                print('\nDataSet %s  - eval_loss: %4f - eval_acc:%4f - eval_f1:%4f - eval_sov:%4f\n'
                      %
                      (args.test_dataset, eval_loss.item() / count,  eval_acc, eval_f1, sov_score / count))
                # logger.info(
                #     '\n\nEpoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - eval_f1:%4f\n'
                #     %
                #     (e + 1, train_loss.item(), eval_loss.item() / count, train_acc, eval_acc, eval_f1))
                print("saving model...\n")
                save_model(model, args.output_dir, step=e+1, acc=eval_acc, loss=eval_loss.item() / count)

                # if train_loss < best_loss:
                #     best_loss = train_loss
                #     # save_model(model, args.output_dir, step=e, f_1=eval_f1)
                #     save_model(model, args.output_dir, step=e, f_1=best_loss)

            if e == 0:
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                # model_module.config.to_json_file(output_config_file)

            if e % verbose == 0:
                if args.is_train == True:
                    train_losses.append(train_loss.item())
                    train_accuracy.append(train_acc)
                eval_losses.append(eval_loss.item() / count)
                eval_accuracy.append(eval_acc)

    # -----------------------predict----------------------------
        if args.is_single == True:
            model.eval()
            with torch.no_grad():
                seq = args.seq
                seq_length = len(seq)
                print('Seq_length:{}\tSeq: {}'.format(seq_length, seq))
                input_ids, output_mask = single_to_features(seq)

                tensors_pt = get_single_embeddings(seq).unsqueeze(0)  # torch.Size([1,108, 1024])
                tensors_seq = model_module.get_embeddings(input_ids.cuda(), None, seq_length)  # torch.Size([1,108, 1024])
                tensors = torch.cat((tensors_pt, tensors_seq), dim=-1)  # torch.Size([1,108, 2048])

                encode = model(tensors)  # torch.Size(1, 108, 3])
                predicts = model_module.predict(encode, output_mask)
                output = get_output(predicts)

                print("PSSP Output: ", output)
                Result = output

    return Result
    # loss_acc_plot(history)
