import pickle
import json

from tqdm import tqdm

import configparser

from scipy import spatial
import numpy as np
import os
from os import sys, path
import random

# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# from embeddings.TransE import *

config = configparser.ConfigParser()
config.read("paths.cfg")


cpnet = None
cpnet_simple = None
concept2id = None
relation2id = None
id2relation = None
id2concept = None
concept_embs = None
relation_embs = None
mcp_py_filenmae = None

# def test():
#     global id2concept, id2relation
#     init_predict(2,5,2)
#     print(id2concept[2])
#     print(id2concept[5])
#     print(id2rel[2])


def load_resources(method):

    global concept2id, id2concept, concept_embs, relation2id, id2relation, relation_embs
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")

    concept_embs = np.load("../embeddings/openke_data/embs/glove_initialized/glove.transe.sgd.ent.npy")

    print("concept_embs done")

    if method == "triple_cls":

        relation2id = {}
        id2relation = {}

        with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
            for w in f.readlines():
                relation2id[w.strip()] = len(relation2id)
                id2relation[len(id2relation)] = w.strip()

        print("relation2id done")

        relation_embs = np.load("../embeddings/openke_data/embs/glove_initialized/glove.transe.sgd.rel.npy")

        print("relation_embs done")

    return


def vanila_score_triple(h, t, r):

    # return np.linalg.norm(t-h-r)
    return (1 + 1 - spatial.distance.cosine(r, t - h)) / 2




def vanila_score_triples(concept_id, relation_id):


    global relation_embs, concept_embs, id2relation, id2concept

    concept = concept_embs[concept_id]
    relation = []

    flag = []
    for i in range(len(relation_id)):

        embs = []
        l_flag = []

        if 0 in relation_id[i] and 17 not in relation_id[i]:
            relation_id[i].append(17)
        elif 17 in relation_id[i] and 0 not in relation_id[i]:
            relation_id[i].append(0)

        if 15 in relation_id[i] and 32 not in relation_id[i]:
            relation_id[i].append(32)
        elif 32 in relation_id[i] and 15 not in relation_id[i]:
            relation_id[i].append(15)


        for j in range(len(relation_id[i])):

            if relation_id[i][j] >= 17:

                score = vanila_score_triple(concept[i + 1], concept[i], relation_embs[relation_id[i][j] - 17])

                print("%s\tr-%s\t%s" % (id2concept[concept_id[i]], id2relation[relation_id[i][j] - 17], id2concept[concept_id[i + 1]]))
                print("Likelihood: " + str(score) + "\n")



            else:

                score = vanila_score_triple(concept[i], concept[i + 1], relation_embs[relation_id[i][j]])

                print("%s\t%s\t%s" % (id2concept[concept_id[i]], id2relation[relation_id[i][j]], id2concept[concept_id[i + 1]]))
                print("Likelihood: " + str(score) + "\n")





def score_triple(h, t, r, flag):

    res = -10

    for i in range(len(r)):
        if flag[i]:
            temp_h, temp_t = t, h
        else:
            temp_h, temp_t = h, t

        # result  = (cosine_sim + 1) / 2
        res = max(res, (1 + 1 - spatial.distance.cosine(r[i], temp_t - temp_h)) / 2)

    return res


def score_triples(concept_id, relation_id, debug=False):

    global relation_embs, concept_embs, id2relation, id2concept

    concept = concept_embs[concept_id]
    relation = []

    flag = []
    for i in range(len(relation_id)):

        embs = []
        l_flag = []

        if 0 in relation_id[i] and 17 not in relation_id[i]:
            relation_id[i].append(17)
        elif 17 in relation_id[i] and 0 not in relation_id[i]:
            relation_id[i].append(0)

        if 15 in relation_id[i] and 32 not in relation_id[i]:
            relation_id[i].append(32)
        elif 32 in relation_id[i] and 15 not in relation_id[i]:
            relation_id[i].append(15)

        for j in range(len(relation_id[i])):

            if relation_id[i][j] >= 17:
                embs.append(relation_embs[relation_id[i][j] - 17])
                l_flag.append(1)

            else:
                embs.append(relation_embs[relation_id[i][j]])
                l_flag.append(0)


        relation.append(embs)

        flag.append(l_flag)


    res = 1

    for i in range(concept.shape[0] - 1):
        h = concept[i]
        t = concept[i + 1]
        score = score_triple(h, t, relation[i], flag[i])

        res *= score

    if debug:
        print("Num of concepts:")
        print(len(concept_id))


        to_print = ""

        for i in range(concept.shape[0] - 1):

            h = id2concept[concept_id[i]]

            to_print += h + "\t"
            for rel in relation_id[i]:
                if rel >= 17:

                    # 'r-' means reverse
                    to_print += ("r-" + id2relation[rel - 17] + "/  ")
                else:
                    to_print += id2relation[rel] + "/  "


        to_print += id2concept[concept_id[-1]]
        print(to_print)

        print("Likelihood: " + str(res) + "\n")

    return res


def context_per_qa(acs, qcs, pooling="mean"):
    '''
    calculate the context embedding for each q-a statement in terms of mentioned concepts
    '''

    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple, concept_embs
    for i in range(len(acs)):
        acs[i] = concept2id[acs[i]]

    for i in range(len(qcs)):
        qcs[i] = concept2id[qcs[i]]

    concept_ids = np.asarray(list(set(qcs) | set(acs)), dtype=int)
    concept_context_emb = np.mean(concept_embs[concept_ids], axis=0) if pooling=="mean" else np.maximum(concept_embs[concept_ids])

    return concept_context_emb


def path_scoring(path, context):

    global concept_embs

    path_concepts = concept_embs[path]


    # cosine distance, the smaller the more alike

    cosine_dist = np.apply_along_axis(spatial.distance.cosine, 1, path_concepts, context)
    cosine_sim = 1 - cosine_dist
    if len(path) > 2:
        return min(cosine_sim[1:-1]) # the minimum of the cos sim of the middle concepts
    else:
        return 1.0 # the source and target of the paths are qa concepts


def calc_context_emb(pooling="mean", filename =""):
    global mcp_py_filenmae
    mcp_py_filenmae = filename + "." + pooling + ".npy"
    if os.path.exists(mcp_py_filenmae):
        print(mcp_py_filenmae, "exists!")
        return

    with open(filename, "rb") as f:
        mcp = json.load(f)

    embs = []

    for s in tqdm(mcp, desc="Computing concept-context embedding.."):
        qcs = s["qc"]
        acs = s["ac"]

        embs.append(context_per_qa(acs=acs, qcs=qcs, pooling=pooling))


    embs = np.asarray(embs)
    print("output_path: " + mcp_py_filenmae)
    np.save(mcp_py_filenmae, embs)




def score_paths(filename, score_filename, method, debug=False, debug_range=None):

    global id2concept, mcp_py_filenmae

    print("Loading paths")

    with open(filename, "rb") as f:
        input = pickle.load(f)

    print("Paths loaded")

    if not method == "triple_cls":

        print("Loading context embeddings")

        context_embs = np.load(mcp_py_filenmae)

        print("Loaded")

    all_scores = []

    if debug:
        a, b =debug_range
        input = input[a:b]
    else:
        pass

    for index, qa_pairs in tqdm(enumerate(input), desc="Scoring the paths", total=len(input)):
        statemetn_scores = []
        for qa_idx, qas in enumerate(qa_pairs):
            statement_paths = qas["pf_res"]

            if statement_paths is not None:

                if not method == "triple_cls":

                    context_emb = context_embs[index]

                path_scores = []
                for pf_idx, item in enumerate(statement_paths):

                    assert len(item["path"]) > 1
                    # vanila_score_triples(concept_id=item["path"], relation_id=item["rel"])

                    if not method == "triple_cls":
                        score = path_scoring(path=item["path"], context=context_emb)

                    else:
                        score = score_triples(concept_id=item["path"], relation_id=item["rel"], debug=debug)
                    path_scores.append(score)
                statemetn_scores.append(path_scores)
            else:
                statemetn_scores.append(None)

        all_scores.append(statemetn_scores)



    if not debug:

        print("saving the path scores")
        with open(score_filename, 'wb') as fp:
            pickle.dump(all_scores, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("done!")

if __name__=="__main__":
    import sys
    flag = sys.argv[1]
    method = "triple_cls" #
    mcp_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp"%flag
    ori_pckle_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.pickle"%flag
    scores_pckle_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.cls.scores.pickle"%flag

    '''to calculate the context embedding for qas'''

    load_resources(method=method)

    if not method == "triple_cls":
        calc_context_emb(filename=mcp_file)
    # score_paths(filename=ori_pckle_file, score_filename=scores_pckle_file, method=method, debug=True, debug_range=(10, 11))

    score_paths(filename=ori_pckle_file, score_filename=scores_pckle_file, method=method, debug=False)

    # score_paths(filename=ori_pckle_file, score_filename=scores_pckle_file, method=method, debug=True,
    #                 debug_range=(11, 12))
