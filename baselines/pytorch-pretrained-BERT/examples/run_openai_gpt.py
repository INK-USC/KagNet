# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

    This script with default values fine-tunes and evaluate a pretrained OpenAI GPT on the RocStories dataset:
        python run_openai_gpt.py \
          --model_name openai-gpt \
          --do_train \
          --do_eval \
          --train_dataset $ROC_STORIES_DIR/cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv \
          --eval_dataset $ROC_STORIES_DIR/cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv \
          --output_dir ../log \
          --train_batch_size 16 \
"""
import argparse
import os
import csv
import json
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer, OpenAIAdam, cached_path

ROCSTORIES_URL = "https://s3.amazonaws.com/datasets.huggingface.co/ROCStories.tar.gz"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def load_csqa_dataset(dataset_path):
    """ Output a list of tuples(question, 1st answer, 2nd answer, 3rd answer, label) """
    with open(dataset_path, "r", encoding="utf8") as f:
        output = []
        for line in tqdm(f):
            csqa_json = json.loads(line)
            output.append((
                csqa_json["question"]["stem"],                  # Question
                csqa_json["question"]["choices"][0]["text"],    # Choice 1
                csqa_json["question"]["choices"][1]["text"],    # Choice 2
                csqa_json["question"]["choices"][2]["text"],    # Choice 3
                ord(csqa_json["answerKey"]) - ord("A")      # Correct choice (0, 1, 2)
            ))
        return output


# def load_rocstories_dataset(dataset_path):
#     """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
#     with open(dataset_path, encoding='utf_8') as f:
#         f = csv.DictReader(f, delimiter="\t")
#         output = []
#         for row in tqdm(f):
#             output.append((
#                 ' '.join([row["InputSentence1"], row["InputSentence2"], row["InputSentence3"], row["InputSentence4"]]),
#                 row["RandomFifthSentenceQuiz1"],
#                 row["RandomFifthSentenceQuiz2"],
#                 int(row["AnswerRightEnding"]) - 1
#                 ))
#
#     print(output[:5])
#     return output

def pre_process_datasets(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(question, 1st answer, 2nd answer, 3rd answer, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, 3, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, 3), dtype=np.int64)
        lm_labels = np.full((n_batch, 3, input_len), fill_value=-1, dtype=np.int64)
        mc_labels = np.zeros((n_batch,), dtype=np.int64)
        for i, (question, answer1, answer2, answer3, mc_label), in enumerate(dataset):
            with_answer1 = [start_token] + question[:cap_length] + [delimiter_token] + answer1[:cap_length] + [clf_token]
            with_answer2 = [start_token] + question[:cap_length] + [delimiter_token] + answer2[:cap_length] + [clf_token]
            with_answer3 = [start_token] + question[:cap_length] + [delimiter_token] + answer3[:cap_length] + [clf_token]
            input_ids[i, 0, :len(with_answer1)] = with_answer1
            input_ids[i, 1, :len(with_answer2)] = with_answer2
            input_ids[i, 2, :len(with_answer3)] = with_answer3
            mc_token_ids[i, 0] = len(with_answer1) - 1
            mc_token_ids[i, 1] = len(with_answer2) - 1
            mc_token_ids[i, 2] = len(with_answer3) - 1
            lm_labels[i, 0, :len(with_answer1) - 1] = with_answer1[1:]
            lm_labels[i, 1, :len(with_answer2) - 1] = with_answer2[1:]
            lm_labels[i, 2, :len(with_answer3) - 1] = with_answer3[1:]
            mc_labels[i] = mc_label
        all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets


def evaluate(model, device, eval_dataloader, desc="Evaluating"):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in tqdm(eval_dataloader, desc=desc):
        batch = tuple(t.to(device) for t in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels = batch
        with torch.no_grad():
            _, mc_loss = model(input_ids, mc_token_ids, lm_labels, mc_labels)
            _, mc_logits = model(input_ids, mc_token_ids)

        mc_logits = mc_logits.detach().cpu().numpy()
        mc_labels = mc_labels.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(mc_logits, mc_labels)

        eval_loss += mc_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    return eval_loss, eval_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='openai-gpt',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    special_tokens = ['_start_', '_delimiter_', '_classify_']
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = OpenAIGPTDoubleHeadsModel.from_pretrained(args.model_name, num_special_tokens=len(special_tokens))
    model.to(device)

    # Load and encode the datasets
    if not args.train_dataset and not args.eval_dataset:
        roc_stories = cached_path(ROCSTORIES_URL)
    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)
    logger.info("Encoding dataset...")
    train_dataset = load_csqa_dataset(args.train_dataset)

    print("Splitting train 90-10 into train-dev.")
    dev_dataset = train_dataset[int(len(train_dataset) * 0.9):]
    train_dataset = train_dataset[:int(len(train_dataset) * 0.9)]
    test_dataset = load_csqa_dataset(args.eval_dataset)
    datasets = (train_dataset, dev_dataset, test_dataset)
    encoded_datasets = tokenize_and_encode(datasets)

    # Compute the mex input length for the Transformer
    max_length = model.config.n_positions // 2 - 2
    input_length = max(
        len(question[:max_length]) + max(
            len(answer1[:max_length]),
            len(answer2[:max_length]),
            len(answer3[:max_length])) + 3
        for dataset in encoded_datasets for question, answer1, answer2, answer3, _ in dataset)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model

    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_length, *special_tokens_ids)
    train_tensor_dataset = tensor_datasets[0]
    dev_tensor_dataset = tensor_datasets[1]
    test_tensor_dataset = tensor_datasets[2]

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    dev_data = TensorDataset(*dev_tensor_dataset)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.train_batch_size)

    test_data = TensorDataset(*test_tensor_dataset)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(train_data) * args.num_train_epochs // args.train_batch_size
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                           lr=args.learning_rate,
                           warmup=args.warmup_proportion,
                           max_grad_norm=args.max_grad_norm,
                           weight_decay=args.weight_decay,
                           t_total=num_train_optimization_steps)

    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        best_dev_accuracy = 0
        test_acc_best_dev = 0
        best_dev_epoch = 0
        no_up = 0
        tqdm_epoch = tqdm(range(args.num_train_epochs), desc="Epoch")
        for epoch in tqdm_epoch:
            model.train()

            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, mc_token_ids, lm_labels, mc_labels = batch
                losses = model(input_ids, mc_token_ids, lm_labels, mc_labels)
                loss = args.lm_coef * losses[0] + losses[1]

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])

            # train_loss, train_accuracy = evaluate(model, device, train_dataloader, desc="Evaluate Train")
            dev_loss, dev_accuracy = evaluate(model, device, dev_dataloader, desc="Evaluate Dev")
            test_loss, test_accuracy = evaluate(model, device, test_dataloader, desc="Evaluate Test")

            train_loss = tr_loss / nb_tr_steps if args.do_train else None

            if dev_accuracy >= best_dev_accuracy:
                # New best model.
                best_dev_accuracy = dev_accuracy
                test_acc_best_dev = test_accuracy
                best_dev_epoch = epoch + 1
                no_up = 0

                # Save the new best model.
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
            else:
                no_up += 1

            tqdm.write("\t ***** Eval results (Epoch %s) *****" % str(epoch + 1))
            # tqdm.write("\t train_accuracy = %s" % str(train_accuracy))
            tqdm.write("\t dev_accuracy = %s" % str(dev_accuracy))
            tqdm.write("")
            tqdm.write("\t best_dev_accuracy = %s" % str(best_dev_accuracy))
            tqdm.write("\t test_acc_best_dev = %s" % str(test_acc_best_dev))
            tqdm.write("\t best_dev_epoch = %s" % str(best_dev_epoch))
            tqdm.write("\t no_up = %s" % str(no_up))
            tqdm.write("")

            if no_up >= 10:
                tqdm_epoch.close()
                break

        # # Load a trained model that you have fine-tuned
        # model_state_dict = torch.load(output_model_file)
        # model = OpenAIGPTDoubleHeadsModel(config)
        # model.load_state_dict(model_state_dict)
        # model.to(device)

    # if args.do_eval:
    #     test_loss, test_accuracy = evaluate(model, device, test_dataloader)
    #     train_loss = tr_loss/nb_tr_steps if args.do_train else None
    #     result = {'test_loss': test_loss,
    #               'test_accuracy': test_accuracy,
    #               'train_loss': train_loss}
    #
    #     output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    #     with open(output_eval_file, "w") as writer:
    #         logger.info("***** Eval results *****")
    #         for key in sorted(result.keys()):
    #             logger.info("  %s = %s", key, str(result[key]))
    #             writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == '__main__':
    main()
