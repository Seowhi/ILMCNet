import json
from util.Logginger import init_logger
import config.args as args

logger = init_logger("model_net", logging_path=args.log_path)


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id, output_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.output_mask = output_mask


class DataProcessor(object):


    def get_train_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as fr:
            lines = []
            for line in fr:
                _line = line.strip('\n')
                lines.append(_line)
            return lines


class MyPro(DataProcessor):


    def _create_example(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            line = json.loads(line)
            text_a = line["source"]
            label = line["target"]
            try:
                assert len(label.split()) == len(text_a.split())
            except:
                logger.info(f'  Error data  \n')
                print(f'{label.split()}, {len(label.split())}')
                print(f'{text_a.split()}, {len(text_a.split())}')
                continue
            example = InputExample(guid=guid, text_a=text_a, label=label)
            examples.append(example)
        return examples

    def get_train_examples(self, path):
        lines = self._read_json(path)
        examples = self._create_example(lines, "train")
        return examples

    def get_dev_examples(self, path):
        lines = self._read_json(path)
        examples = self._create_example(lines, "dev")
        return examples

    def get_labels(self):
        return args.labels


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    index = 0
    for ex_index, example in enumerate(examples):
        index = ex_index
        tokens_a = tokenizer.tokenize(example.text_a)
        labels = example.label.split()

        if len(tokens_a) == 0 or len(labels) == 0:
            continue

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            labels = labels[:(max_seq_length - 2)]

        ex_id = args.train_dataset + "_" + str(ex_index)
        # tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens = [ex_id] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        label_id = [label_map.get(l, len(label_map) - 1) for l in labels]
        label_padding = [-1] * (max_seq_length - len(label_id))
        label_id += label_padding


        output_mask = [0] + len(tokens_a) * [1] + [0]
        output_mask += padding

        # if ex_index < 1:
        #     logger.info("-----------------Example-----------------")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("label: %s " % " ".join([str(x) for x in label_id]))
        #     logger.info("output_mask: %s " % " ".join([str(x) for x in output_mask]))
        # ----------------------------------------------------


        feature = InputFeature(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label_id=label_id,
                               output_mask=output_mask)
        features.append(feature)

    # print(index)
    # exit()
    return features

