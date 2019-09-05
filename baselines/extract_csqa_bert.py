import argparse
import csv
import json
import logging
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

sys.path.append("./pytorch-pretrained-BERT")
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForMultipleChoice, BertForMultipleChoiceExtraction, WEIGHTS_NAME, CONFIG_NAME, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

'''
python extract_csqa_bert.py --bert_model bert-large-uncased --do_eval --do_lower_case --data_dir ../datasets/csqa_new --eval_batch_size 60 --learning_rate 1e-4  --max_seq_length 60 --output_dir ./models/ --save_model_name bert_large_1e-4_g4w05wd05m70 --data_split_to_extract dev_rand_split.jsonl --output_sentvec_file ../datasets/csqa_new/dev_rand_split.jsonl.statements.fintuned.large.npy
'''

class SwagExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 swag_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 ending_4,
                 label=None):
        self.swag_id = swag_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
            ending_4,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "swag_id: {}".format(self.swag_id),
            "context_sentence: {}".format(self.context_sentence),
            "start_ending: {}".format(self.start_ending),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2]),
            "ending_3: {}".format(self.endings[3]),
            "ending_4: {}".format(self.endings[4]),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def read_csqa_examples(input_file, have_answer=True):
    with open(input_file, "r", encoding="utf-8") as f:
        examples = []
        for line in f.readlines():
            csqa_json = json.loads(line)
            if have_answer:
                label = ord(csqa_json["answerKey"]) - ord("A")
            else:
                label = 0  # just as placeholder here for the test data
            examples.append(
                SwagExample(
                    swag_id=csqa_json["id"],
                    context_sentence=csqa_json["question"]["stem"],
                    start_ending="",
                    ending_0=csqa_json["question"]["choices"][0]["text"],
                    ending_1=csqa_json["question"]["choices"][1]["text"],
                    ending_2=csqa_json["question"]["choices"][2]["text"],
                    ending_3=csqa_json["question"]["choices"][3]["text"],
                    ending_4=csqa_json["question"]["choices"][4]["text"],
                    label = label
                ))
    return examples



def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index == 0 and False:
            logger.info("*** Example ***")
            logger.info("swag_id: {}".format(example.swag_id))
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
            if is_training:
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id = example.swag_id,
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def get_eval_dataloader(eval_features, args):
    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    return eval_dataloader


def get_train_dataloader(train_features, args):
    all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader


def evaluate(model, device, eval_dataloader, desc="Evaluate"):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc=desc):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    return eval_loss, eval_accuracy


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--output_sentvec_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file of extracted embedding files of sentences.")

    parser.add_argument("--data_split_to_extract",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file of extracted embedding files of sentences.")


    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--epoch_id",
                        default=0,
                        type=int,
                        help="Epoch id to extract.")

    parser.add_argument("--save_model_name",
                        default="model",
                        type=str,
                        required=True,
                        help="The output model name where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--with_dev",
                        action='store_true',
                        help="Whether to run training with dev.")


    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test on the test set.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--layer_id",
                        default=-1,
                        type=int,
                        help="Output Layer Id")

    parser.add_argument("--mlp_hidden_dim",
                        default=64,
                        type=int,
                        help="mlp_hidden_dim.")

    parser.add_argument("--mlp_dropout",
                        default=0.1,
                        type=float,
                        help="hidden drop out")


    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--patience',
                        type=int,
                        default=5,
                        help="early stop epoch nums on dev")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")



    args = parser.parse_args()
    print("torch.cuda.is_available()", torch.cuda.is_available())
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_csqa_examples(os.path.join(args.data_dir, 'train_rand_split.jsonl'))
        dev_examples = read_csqa_examples(os.path.join(args.data_dir, 'dev_rand_split.jsonl'))
        print(len(train_examples))
        if args.with_dev:
            train_examples += dev_examples
            print(len(train_examples))

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = BertForMultipleChoiceExtraction.from_pretrained(args.bert_model,
        cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank)),
        num_choices=5, mlp_hidden_dim=args.mlp_hidden_dim, mlp_dropout=args.mlp_dropout)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0

    model.to(device)


    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Load a trained model and config that you have fine-tuned
        output_model_file = os.path.join(args.output_dir, args.save_model_name + ".bin.%d"%(args.epoch_id))
        output_config_file = os.path.join(args.output_dir, args.save_model_name + ".config")
        config = BertConfig(output_config_file)
        model = BertForMultipleChoiceExtraction(config, num_choices=5, mlp_hidden_dim=args.mlp_hidden_dim, mlp_dropout=args.mlp_dropout)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)
        # to extract dev_rand_split.jsonl 'dev_rand_split.jsonl'
        eval_examples = read_csqa_examples(os.path.join(args.data_dir, args.data_split_to_extract))

        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        pooled_sent_vecs = []
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Iteration"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                # tmp_eval_loss, pooled_output = model(input_ids, segment_ids, input_mask, label_ids)
                logits, pooled_output = model(input_ids, segment_ids, input_mask, layer_id = args.layer_id)
            pooled_sent_vecs.append(pooled_output)
            # print(pooled_output.size())
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()


            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        pooled_sent_vecs = torch.cat(pooled_sent_vecs, dim=0)
        print(pooled_sent_vecs.size())
        output_numpy = pooled_sent_vecs.to('cpu').numpy()
        print(output_numpy.shape)

        np.save(args.output_sentvec_file+".%d"%(args.layer_id), output_numpy)



        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_accuracy': eval_accuracy}
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))


if __name__ == "__main__":
    print("torch.cuda.is_available()", torch.cuda.is_available())
    main()
