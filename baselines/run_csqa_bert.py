import argparse
import csv
import json
import logging
import os
import random
import sys
from io import open
import re
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# String used to indicate a blank
BLANK_STR = "___"
sys.path.append("./pytorch-pretrained-BERT")
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForMultipleChoice, WEIGHTS_NAME, CONFIG_NAME, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# Get a Fill-In-The-Blank (FITB) statement from the question text. E.g. "George wants to warm his
# hands quickly by rubbing them. Which skin surface will produce the most heat?" ->
# "George wants to warm his hands quickly by rubbing them. ___ skin surface will produce the most
# heat?
def get_fitb_from_question(question_text: str) -> str:
    fitb = replace_wh_word_with_blank(question_text)
    if not re.match(".*_+.*", fitb):
        # print("Can't create hypothesis from: '{}'. Appending {} !".format(question_text, BLANK_STR))
        # Strip space, period and question mark at the end of the question and add a blank
        fitb = re.sub("[\.\? ]*$", "", question_text.strip()) + " "+ BLANK_STR
    return fitb


# Create a hypothesis statement from the the input fill-in-the-blank statement and answer choice.
def create_hypothesis(fitb: str, choice: str) -> str:
    if ". " + BLANK_STR in fitb or fitb.startswith(BLANK_STR):
        choice = choice[0].upper() + choice[1:]
    else:
        choice = choice.lower()
    # Remove period from the answer choice, if the question doesn't end with the blank
    if not fitb.endswith(BLANK_STR):
        choice = choice.rstrip(".")
    # Some questions already have blanks indicated with 2+ underscores
    hypothesis = re.sub("__+", choice, fitb)
    return hypothesis


# Identify the wh-word in the question and replace with a blank
def replace_wh_word_with_blank(question_str: str):
    if "What is the name of the government building that houses the U.S. Congress?" in question_str:
        print()
    question_str = question_str.replace("What's", "What is")
    question_str = question_str.replace("whats", "what")
    question_str = question_str.replace("U.S.", "US")
    wh_word_offset_matches = []
    wh_words = ["which", "what", "where", "when", "how", "who", "why"]
    for wh in wh_words:
        # Some Turk-authored SciQ questions end with wh-word
        # E.g. The passing of traits from parents to offspring is done through what?

        if wh == "who" and "people who" in question_str:
            continue

        m = re.search(wh + "\?[^\.]*[\. ]*$", question_str.lower())
        if m:
            wh_word_offset_matches = [(wh, m.start())]
            break
        else:
            # Otherwise, find the wh-word in the last sentence
            m = re.search(wh + "[ ,][^\.]*[\. ]*$", question_str.lower())
            if m:
                wh_word_offset_matches.append((wh, m.start()))
            # else:
            #     wh_word_offset_matches.append((wh, question_str.index(wh)))

    # If a wh-word is found
    if len(wh_word_offset_matches):
        # Pick the first wh-word as the word to be replaced with BLANK
        # E.g. Which is most likely needed when describing the change in position of an object?
        wh_word_offset_matches.sort(key=lambda x: x[1])
        wh_word_found = wh_word_offset_matches[0][0]
        wh_word_start_offset = wh_word_offset_matches[0][1]
        # Replace the last question mark with period.
        question_str = re.sub("\?$", ".", question_str.strip())
        # Introduce the blank in place of the wh-word
        fitb_question = (question_str[:wh_word_start_offset] + BLANK_STR +
                         question_str[wh_word_start_offset + len(wh_word_found):])
        # Drop "of the following" as it doesn't make sense in the absence of a multiple-choice
        # question. E.g. "Which of the following force ..." -> "___ force ..."
        final = fitb_question.replace(BLANK_STR + " of the following", BLANK_STR)
        final = final.replace(BLANK_STR + " of these", BLANK_STR)
        return final

    elif " them called?" in question_str:
        return question_str.replace(" them called?", " " + BLANK_STR+".")
    elif " meaning he was not?" in question_str:
        return question_str.replace(" meaning he was not?", " he was not " + BLANK_STR+".")
    elif " one of these?" in question_str:
        return question_str.replace(" one of these?", " " + BLANK_STR+".")
    elif re.match(".*[^\.\?] *$", question_str):
        # If no wh-word is found and the question ends without a period/question, introduce a
        # blank at the end. e.g. The gravitational force exerted by an object depends on its
        return question_str + " " + BLANK_STR
    else:
        # If all else fails, assume "this ?" indicates the blank. Used in Turk-authored questions
        # e.g. Virtually every task performed by living organisms requires this?
        return re.sub(" this[ \?]", " ___ ", question_str)



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


# the original one
def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # CSQA is a multiple choice task. To perform this task using Bert,
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
    # - [CLS] context [SEP] choice_5 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):

        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens

            statement = create_hypothesis(get_fitb_from_question(example.context_sentence), ending)

            statement = example.context_sentence

            context_tokens = tokenizer.tokenize(statement)
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


def m1c_convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # CSQA is a multiple choice task. To perform this task using Bert,
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP] choice_2 [SEP] choice_3 [SEP] choice_4 [SEP] choice_5 [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_1 [SEP] choice_2 [SEP] choice_3 [SEP] choice_4 [SEP] choice_5 [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_1 [SEP] choice_2 [SEP] choice_3 [SEP] choice_4 [SEP] choice_5 [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_1 [SEP] choice_2 [SEP] choice_3 [SEP] choice_4 [SEP] choice_5 [SEP] choice_4 [SEP]
    # - [CLS] context [SEP] choice_1 [SEP] choice_2 [SEP] choice_3 [SEP] choice_4 [SEP] choice_5 [SEP] choice_5 [SEP]
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

            all_ending_tokens_list = [start_ending_tokens + tokenizer.tokenize(cur_ending)
                                    for cur_ending_index, cur_ending in enumerate(example.endings)
                                      if cur_ending_index != ending_index
                                      ]
            assert len(all_ending_tokens_list) == len(example.endings) - 1

            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            all_ending_tokens = []
            for cur_ind, et in enumerate(all_ending_tokens_list):
                all_ending_tokens = all_ending_tokens + ["[SEP]"] + et


            context_tokens_choice = context_tokens_choice + all_ending_tokens

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
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

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

    parser.add_argument("--wsc",
                        action='store_true',
                        help="Whether to run training with wsc.")

    parser.add_argument("--swag_transfer",
                        action='store_true',
                        help="Whether to run training with swag.")

    parser.add_argument("--inhouse",
                        action='store_true',
                        help="Whether to run eval on the inhouse train/dev set.")

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

    parser.add_argument("--epoch_suffix",
                        default=0,
                        type=int,
                        help="Epoch suffix number.")

    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")


    parser.add_argument("--mlp_hidden_dim",
                        default=64,
                        type=int,
                        help="mlp_hidden_dim.")

    parser.add_argument("--mlp_dropout",
                        default=0.1,
                        type=float,
                        help="hidden drop out")

    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight decay for optimization")

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
        ori_train_examples = read_csqa_examples(os.path.join(args.data_dir, 'train_rand_split.jsonl'))
        ori_dev_examples = read_csqa_examples(os.path.join(args.data_dir, 'dev_rand_split.jsonl'))
        ori_test_examples = read_csqa_examples(os.path.join(args.data_dir, 'train2_rand_split.jsonl'))

        if args.inhouse:
            train_examples = ori_train_examples[0:850]  #8500
            test_examples = ori_train_examples[8500:]
            dev_examples = ori_dev_examples[:]
        else:
            train_examples = ori_train_examples[:]
            dev_examples = ori_dev_examples[:]

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = BertForMultipleChoice.from_pretrained(args.bert_model,
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
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
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
    if args.do_train:
        train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length, True)
        dev_features = convert_examples_to_features(dev_examples, tokenizer, args.max_seq_length, True)

        # dev_features = train_features[int(len(train_features) * 0.9):]

        # random.shuffle(train_features)
        # train_features = train_features[:int(len(train_features) * 0.8)]

        train_dataloader = get_train_dataloader(train_features, args)
        dev_dataloader = get_eval_dataloader(dev_features, args)
        if args.inhouse:
            test_features = convert_examples_to_features(test_examples, tokenizer, args.max_seq_length, True)
            test_dataloader = get_eval_dataloader(test_features, args)


        # test_examples = read_csqa_examples(os.path.join(args.data_dir, 'dev_rand_split.jsonl'))
        # test_features = convert_examples_to_features(
        #     test_examples, tokenizer, args.max_seq_length, True)


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        logger.info("")
        logger.info("  Num train features = %d", len(train_features))
        logger.info("  Num dev features = %d", len(dev_features))
        # logger.info("  Num test features = %d", len(test_features))

        best_dev_accuracy = 0
        best_dev_epoch = 0
        no_up = 0

        epoch_tqdm = trange(int(args.num_train_epochs), desc="Epoch")
        for epoch in epoch_tqdm:
            model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # train_loss, train_accuracy = evaluate(model, device, train_dataloader, desc="Evaluate Train")
            dev_loss, dev_accuracy = evaluate(model, device, dev_dataloader, desc="Evaluate Dev")
            if args.inhouse:
                test_loss, test_accuracy = evaluate(model, device, test_dataloader, desc="Evaluate Test")
            # test_loss, test_accuracy = dev_loss, dev_accuracy

            if dev_accuracy > best_dev_accuracy:
                # New best model.
                best_dev_accuracy = dev_accuracy
                best_dev_epoch = epoch + 1
                no_up = 0

                # Save the new best model.
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, args.save_model_name+".bin.%d"%(epoch))
                torch.save(model_to_save.state_dict(), output_model_file)

                output_config_file = os.path.join(args.output_dir, args.save_model_name+".config")
                with open(output_config_file, 'w') as fpp:
                     fpp.write(model_to_save.config.to_json_string())


            else:
                no_up += 1

            tqdm.write("\t ***** Eval results (Epoch %s) *****" % str(epoch + 1))
            # tqdm.write("\t train_accuracy = %s" % str(train_accuracy))
            tqdm.write("\t dev_accuracy = %s" % str(dev_accuracy))
            tqdm.write("")
            if args.inhouse:
                tqdm.write("\t test_accuracy = %s" % str(test_accuracy))
                tqdm.write("")
            tqdm.write("\t best_dev_accuracy = %s" % str(best_dev_accuracy))
            tqdm.write("\t best_dev_epoch = %s" % str(best_dev_epoch))
            tqdm.write("\t no_up = %s" % str(no_up))
            tqdm.write("")

            if no_up >= args.patience:
                epoch_tqdm.close()
                break

    model.to(device)


    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Load a trained model and config that you have fine-tuned
        output_model_file = os.path.join(args.output_dir, args.save_model_name + ".bin.%d"%(args.epoch_suffix))
        output_config_file = os.path.join(args.output_dir, args.save_model_name + ".config")
        config = BertConfig(output_config_file)
        model = BertForMultipleChoice(config, num_choices=5, mlp_hidden_dim=args.mlp_hidden_dim, mlp_dropout=args.mlp_dropout)

        model.load_state_dict(torch.load(output_model_file))
        model.to(device)

        if args.wsc:
            eval_examples = read_csqa_examples('../datasets/wsc.jsonl') # for testing on wsc
        elif args.swag_transfer:
            eval_examples = read_csqa_examples('../datasets/swagaf/data/val.jsonl')  # for testing on wsc
        else:
            eval_examples = read_csqa_examples(os.path.join(args.data_dir, 'dev_rand_split.jsonl'))

        eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, True)

        if args.inhouse:
            # train_examples = read_csqa_examples(os.path.join(args.data_dir, 'train_rand_split.jsonl'))
            # eval_examples = train_examples[8500:]
            # eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, True)

            eval_examples_test = read_csqa_examples(os.path.join(args.data_dir, 'train_rand_split.jsonl'))[8500:]
            eval_features_test = convert_examples_to_features(eval_examples_test, tokenizer, args.max_seq_length, True)

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
        test_outputs = []
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            outputs = np.argmax(logits, axis=1)
            test_outputs += list(outputs)
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        if args.wsc:
            result = {'eval_accuracy': eval_accuracy}
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            test_output_file = os.path.join(args.output_dir, args.save_model_name + "_wsc_prediction.csv")
            with open(test_output_file, 'w') as fout:
                with open(os.path.join('../datasets/wsc.jsonl'), "r", encoding="utf-8") as fin:
                    examples = []
                    for i, line in enumerate(fin.readlines()):
                        csqa_json = json.loads(line)
                        label_pred = chr(ord("A") + test_outputs[i])
                        if label_pred in ["C", "E"]:
                            label_pred = "A"
                        if label_pred in ["D"]:
                            label_pred = "B"
                        fout.write(csqa_json["id"] + "," + str(label_pred) + "\n")
            
        elif args.swag_transfer:
            result = {'eval_accuracy': eval_accuracy}
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            test_output_file = os.path.join(args.output_dir, args.save_model_name + "_swag_val.csv")
            with open(test_output_file, 'w') as fout:
                with open(os.path.join('../datasets/swagaf/data/val.jsonl'), "r", encoding="utf-8") as fin:
                    examples = []
                    for i, line in enumerate(fin.readlines()):
                        csqa_json = json.loads(line)
                        label_pred = chr(ord("A") + test_outputs[i])
                        if label_pred == "E":
                            label_pred = "A"
                        fout.write(csqa_json["id"] + "," + str(label_pred) + "\n")

        elif args.inhouse:
            dev_result = {'dev_eval_accuracy': eval_accuracy}

            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor(select_field(eval_features_test, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(eval_features_test, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(eval_features_test, 'segment_ids'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in eval_features_test], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            test_outputs = []
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                outputs = np.argmax(logits, axis=1)
                test_outputs += list(outputs)
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            test_result = {'test_eval_accuracy': eval_accuracy}
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        else:
            result = {'eval_accuracy': eval_accuracy}
            test_output_file = os.path.join(args.output_dir, args.save_model_name + "_dev_output.csv")
            with open(test_output_file, 'w') as fout:
                with open(os.path.join(args.data_dir, 'dev_rand_split.jsonl'), "r", encoding="utf-8") as fin:
                    examples = []
                    for i, line in enumerate(fin.readlines()):
                        csqa_json = json.loads(line)
                        label_pred = chr(ord("A") + test_outputs[i])
                        fout.write(csqa_json["id"] + "," + str(label_pred) + "\n")


            output_eval_file = os.path.join(args.output_dir, args.save_model_name + "_res_on_dev.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

    if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Load a trained model and config that you have fine-tuned
        output_model_file = os.path.join(args.output_dir, args.save_model_name + ".bin.%d"%(args.epoch_suffix))
        output_config_file = os.path.join(args.output_dir, args.save_model_name + ".config")
        config = BertConfig(output_config_file)
        # model = BertForMultipleChoice(config, num_choices=5)
        model = BertForMultipleChoice(config, num_choices=5, mlp_hidden_dim=args.mlp_hidden_dim, mlp_dropout=args.mlp_dropout)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)
        eval_examples = read_csqa_examples(os.path.join(args.data_dir, 'test_rand_split_no_answers.jsonl'), have_answer=False)
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
        test_outputs = []
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            # label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            # print(logits.shape)
            # print(logits)
            outputs = np.argmax(logits, axis=1)
            # break
            # print(outputs)
            test_outputs += list(outputs)
            # print(test_outputs)


        test_output_file = os.path.join(args.output_dir, args.save_model_name + "_test_output.csv")
        with open(test_output_file, 'w') as fout:
            with open(os.path.join(args.data_dir, 'test_rand_split_no_answers.jsonl'), "r", encoding="utf-8") as fin:
                examples = []
                for i, line in enumerate(fin.readlines()):
                    csqa_json = json.loads(line)
                    label_pred = chr(ord("A")+test_outputs[i])
                    fout.write(csqa_json["id"]+","+str(label_pred)+"\n")



if __name__ == "__main__":
    print("torch.cuda.is_available()", torch.cuda.is_available())
    main()
