import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from torch.nn import init
import torch.utils.data as data
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
from models import KnowledgeEnhancedRelationNetwork, RelationNetwork, weight_init, GCN_Sent, KnowledgeAwareGraphNetworks
from tqdm import tqdm
from csqa_dataset import data_with_paths, collate_csqa_paths, data_with_graphs, data_with_graphs_and_paths, collate_csqa_graphs, collate_csqa_graphs_and_paths
from parallel import DataParallelModel, DataParallelCriterion
import copy
import random
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def load_embeddings(pretrain_embed_path):
    print("Loading glove concept embeddings with pooling:", pretrain_embed_path)
    concept_vec = np.load(pretrain_embed_path)
    print("done!")
    return concept_vec


def train_epoch_kag_netowrk(train_set, batch_size, optimizer, device, model, num_choice, loss_func):
    model.train()
    dataset_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True,
                                     collate_fn=collate_csqa_graphs_and_paths)
    bce_loss_func = nn.BCELoss()
    # bce_loss_func = DataParallelCriterion(bce_loss_func)
    for k, (statements, correct_labels, graphs, cpt_paths, rel_paths, qa_pairs, concept_mapping_dicts) in enumerate(
            tqdm(dataset_loader, desc="Train Batch")):
        optimizer.zero_grad()
        statements = statements.to(device)
        correct_labels = correct_labels.to(device)
        graphs.ndata['cncpt_ids'] = graphs.ndata['cncpt_ids'].to(device)
        flat_statements = []  # flat to ungroup the questions
        flat_qa_pairs = []
        flat_cpt_paths = []
        flat_rel_paths = []
        assert len(statements) == len(cpt_paths) == len(rel_paths) == len(qa_pairs)
        for i in range(len(statements)):
            cur_statement = statements[i][0]  # num_choice statements
            cur_qa_pairs = qa_pairs[i]
            cur_cpt_paths = cpt_paths[i]
            cur_rel_paths = rel_paths[i]

            flat_statements.extend(cur_statement)
            flat_qa_pairs.extend(cur_qa_pairs)
            flat_cpt_paths.extend(cur_cpt_paths)
            flat_rel_paths.extend(cur_rel_paths)

        flat_statements = torch.stack(flat_statements).to(device)
        flat_logits = model(flat_statements, flat_qa_pairs, flat_cpt_paths, flat_rel_paths, graphs, concept_mapping_dicts)


        y = torch.Tensor([1] * len(statements) * (num_choice - 1)).to(device)
        assert len(flat_logits) == len(flat_statements)
        assert len(flat_statements) == len(statements) * num_choice
        x1 = []
        x2 = []

        for j, correct in enumerate(correct_labels):
            # for a particular qeustion
            for i in range(num_choice):
                cur_logit = flat_logits[j * num_choice + i]
                if i != correct[0]:  # for wrong answers
                    x2.append(cur_logit)
                else:  # for the correct answer
                    for _ in range(num_choice - 1):
                        x1.append(cur_logit)
        mrloss = loss_func(torch.cat(x1), torch.cat(x2), y)  # margin ranking loss
        mrloss.backward()
        optimizer.step()



def eval_kag_netowrk(eval_set, batch_size ,  device, model, num_choice):
    model.eval()
    dataset_loader = data.DataLoader(eval_set, batch_size=batch_size, num_workers=0, shuffle=True,
                                     collate_fn=collate_csqa_graphs_and_paths)
    cnt_correct = 0
    for k, (statements, correct_labels, graphs, cpt_paths, rel_paths, qa_pairs, concept_mapping_dicts) in enumerate(
            tqdm(dataset_loader, desc="Eval Batch")):
        statements = statements.to(device)
        correct_labels = correct_labels.to(device)
        graphs.ndata['cncpt_ids'] = graphs.ndata['cncpt_ids'].to(device)
        flat_statements = []  # flat to ungroup the questions
        flat_qa_pairs = []
        flat_cpt_paths = []
        flat_rel_paths = []
        assert len(statements) == len(cpt_paths) == len(rel_paths) == len(qa_pairs)
        for i in range(len(statements)):
            cur_statement = statements[i][0]  # num_choice statements
            cur_qa_pairs = qa_pairs[i]
            cur_cpt_paths = cpt_paths[i]
            cur_rel_paths = rel_paths[i]

            flat_statements.extend(cur_statement)
            flat_qa_pairs.extend(cur_qa_pairs)
            flat_cpt_paths.extend(cur_cpt_paths)
            flat_rel_paths.extend(cur_rel_paths)

        flat_statements = torch.stack(flat_statements).to(device)
        flat_logits = model(flat_statements, flat_qa_pairs, flat_cpt_paths, flat_rel_paths, graphs,
                            concept_mapping_dicts)


        assert len(flat_statements) == len(statements) * num_choice

        # flat_loss = loss_function(label_preds, labels)
        for j, correct in enumerate(correct_labels):
            # for a particular qeustion
            max_logit = None
            pred = 0
            for i in range(num_choice):
                cur_logit = flat_logits[j * num_choice + i]
                if max_logit is None:
                    max_logit = cur_logit
                    pred = i
                if max_logit < cur_logit:
                    max_logit = cur_logit
                    pred = i

            if correct[0] == pred:
                cnt_correct += 1
    acc = cnt_correct / len(eval_set)
    return acc





def train_kagnet_main():
    pretrain_cpt_emd_path = "../embeddings/openke_data/embs/glove_initialized/ent.npy"
    pretrain_rel_emd_path = "../embeddings/openke_data/embs/glove_initialized/rel.npy"

    pretrained_concept_emd = load_embeddings(pretrain_cpt_emd_path)
    pretrained_relation_emd = load_embeddings(pretrain_rel_emd_path)
    print("pretrained_concept_emd.shape:", pretrained_concept_emd.shape)
    print("pretrained_relation_emd.shape:", pretrained_relation_emd.shape)

    # add one concept vec for dummy concept
    concept_dim = pretrained_concept_emd.shape[1]
    concept_num = pretrained_concept_emd.shape[0] + 1  # for dummy concept
    pretrained_concept_emd = np.insert(pretrained_concept_emd, 0, np.zeros((1, concept_dim)), 0)

    relation_num = pretrained_relation_emd.shape[0] * 2 + 1  # for inverse and dummy relations
    relation_dim = pretrained_relation_emd.shape[1]
    pretrained_relation_emd = np.concatenate((pretrained_relation_emd, pretrained_relation_emd))
    pretrained_relation_emd = np.insert(pretrained_relation_emd, 0, np.zeros((1, relation_dim)), 0)

    pretrained_concept_emd = torch.FloatTensor(pretrained_concept_emd)
    pretrained_relation_emd = torch.FloatTensor(pretrained_relation_emd)  # torch.FloatTensor(pretrained_relation_emd)

    lstm_dim = 128
    lstm_layer_num = 1
    dropout = 0.0
    bidirect = False
    batch_size = 50
    n_epochs = 15
    num_choice = 5
    sent_dim = 1024
    qas_encoded_dim = 128
    num_random_paths = None
    graph_hidden_dim = 50
    graph_output_dim = 25
    patience = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set = data_with_graphs_and_paths("../datasets/csqa_new/train_rand_split.jsonl.statements",
                      "../datasets/csqa_new/train_rand_split.jsonl.statements.pruned.0.15.pnxg",
                      "../datasets/csqa_new/train_rand_split.jsonl.statements.mcp.pf.cls.pruned.0.15.pickle",
                      "../datasets/csqa_new/train_rand_split.jsonl.statements.finetuned.large.-2.npy",
                      num_choice=5, reload=False, cut_off=3, start=0, end=None)
    

    dev_set = data_with_graphs_and_paths("../datasets/csqa_new/dev_rand_split.jsonl.statements",
                      "../datasets/csqa_new/dev_rand_split.jsonl.statements.pruned.0.15.pnxg",
                      "../datasets/csqa_new/dev_rand_split.jsonl.statements.mcp.pf.cls.pruned.0.15.pickle",
                      "../datasets/csqa_new/dev_rand_split.jsonl.statements.finetuned.large.-2.npy",
                      num_choice=5, reload=False, cut_off=3, start=0, end=None)


    print("len(train_set):", len(train_set), "len(dev_set):", len(dev_set))

    model = KnowledgeAwareGraphNetworks(sent_dim, concept_dim, relation_dim,
                                             concept_num, relation_num, qas_encoded_dim,
                                             pretrained_concept_emd, pretrained_relation_emd,
                                             lstm_dim, lstm_layer_num, device, graph_hidden_dim, graph_output_dim,
                                             dropout=dropout, bidirect=bidirect, num_random_paths=num_random_paths,
                                             path_attention=True, qa_attention=True)
    model.to(device)

    print("checking model parameters")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Trainable: ", name, param.size())
        else:
            print("Fixed: ", name, param.size())  # , param.data)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model num para#:", num_params)

    parameters = filter(lambda p: p.requires_grad, model.parameters())


    optimizer = torch.optim.Adam(parameters, lr=0.001, weight_decay=0.0001, amsgrad=True)
    loss_func = torch.nn.MarginRankingLoss(margin=0.2, size_average=None, reduce=None, reduction='mean')

    no_up = 0
    best_dev_acc = 0.0
    for i in range(n_epochs):
        print('epoch: %d start!' % i)
        train_epoch_kag_netowrk(train_set, batch_size, optimizer, device, model, num_choice, loss_func)

        # train_acc = eval_kag_netowrk(train_set, batch_size, device, model, num_choice)
        # print("training acc: %.5f" % train_acc, end="\t\t")

        dev_acc = eval_kag_netowrk(dev_set, batch_size, device, model, num_choice)
        print("dev acc: %.5f" % dev_acc)

        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            no_up = 0
            torch.save(model.state_dict(),
                       'model_save/{:s}_model_acc_{:.4f}.model'
                       .format("tmp", best_dev_acc))
        else:
            no_up += 1
            if no_up > patience:
                break


train_kagnet_main()
