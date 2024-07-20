import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from preprocessing.data_processor import MyPro, convert_examples_to_features
import config.args as args
from util.Logginger import init_logger

logger = init_logger(f"{args.task_name}", logging_path=args.log_path)


def init_parameters():
    tokenizer = BertTokenizer(vocab_file=args.VOCAB_FILE)
    tokenizer.save_vocabulary(args.output_dir)
    processor = MyPro()
    return tokenizer, processor


def single_to_features(seq):
    tokenizer, processor = init_parameters()
    input_ids = tokenizer.convert_tokens_to_ids(seq.lower())
    output_mask = [1] * len(input_ids)
    output_mask = torch.tensor(output_mask).unsqueeze(0)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    return input_ids, output_mask


def create_batch_iter(mode, path):

    # logger.info(f'{mode} path is {path}')
    tokenizer, processor = init_parameters()
    if mode == "train":
        examples = processor.get_train_examples(path)
        num_train_steps = int(
            len(examples) / args.train_batch_size / args.gradient_accumulation_steps *
            args.num_train_epochs)
        batch_size = args.train_batch_size
        # logger.info("  Num train steps = %d", num_train_steps)
        max_length = args.max_seq_length
    elif mode == "dev":
        examples = processor.get_dev_examples(path)
        batch_size = args.eval_batch_size
        max_length = args.max_valid_length
    else:
        raise ValueError("Invalid mode %s" % mode)

    label_list = processor.get_labels()


    features = convert_examples_to_features(examples, label_list, max_length, tokenizer)
    # logger.info(f"  Num {mode} features = %d", len(features))
    # logger.info(f" {mode} Batch size = %d", batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)

    # print(all_input_ids[0])
    # print(all_input_mask[0])
    # exit()

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                         all_output_mask)

    if mode == "train":
        sampler = RandomSampler(data)
    elif mode == "dev":
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)


    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)
    if mode == "train":
        return iterator, num_train_steps
    elif mode == "dev":
        return iterator
    else:
        raise ValueError("Invalid mode %s" % mode)
