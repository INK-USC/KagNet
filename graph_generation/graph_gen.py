import configparser
import networkx as nx
import itertools
import math
import random
import json
from tqdm import tqdm
import sys
import time
import timeit
import pickle


import sys

split = sys.argv[1]

config = configparser.ConfigParser()
config.read("paths.cfg")

GRAPH_PATH = "../datasets/csqa_new/%s_rand_split.jsonl.statements.pruned.0.15.pnxg"%split
PF_PATH = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.cls.pruned.0.15.pickle"%split
MCP_PATH = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp"%split

NUM_CHOICES = 5

cpnet = None
cpnet_simple = None
concept2id = None
relation2id = None
id2relation = None
id2concept = None
mcp_data = None
pf_data = None


def load_resources():
    global concept2id, relation2id, id2relation, id2concept, mcp_data, pf_data, PF_PATH, MCP_PATH
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)
    print("relation2id done")

    print("loading pf_data from %s" % PF_PATH)
    start_time = timeit.default_timer()
    with open(PF_PATH, "rb") as fi:
        pf_data = pickle.load(fi)
    print('\t Done! Time: ', "{0:.2f} sec".format(float(timeit.default_timer() - start_time)))

    with open(MCP_PATH, "r") as f:
        mcp_data = json.load(f)



def load_cpnet():
    global cpnet,concept2id, relation2id, id2relation, id2concept, cpnet_simple
    print("loading cpnet....")
    cpnet = nx.read_gpickle(config["paths"]["conceptnet_en_graph"])
    print("Done")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def get_edge(src_concept, tgt_concept):
    global cpnet, concept2id, relation2id, id2relation, id2concept
    rel_list = cpnet[src_concept][tgt_concept]
    # tmp = [rel_list[item]["weight"] for item in rel_list]
    # s = tmp.index(min(tmp))
    # rel = rel_list[s]["rel"]
    return list(set([rel_list[item]["rel"] for item in rel_list]))


# plain graph generation
def plain_graph_generation(qcs, acs, paths, rels):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
    # print("qcs", qcs)
    # print("acs", acs)
    # print("paths", paths)
    # print("rels", rels)

    graph = nx.Graph()
    for index, p in enumerate(paths):

        for c_index in range(len(p)-1):
            h = p[c_index]
            t = p[c_index+1]
            # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
            graph.add_edge(h,t, weight=1.0)

    for qc1, qc2 in list(itertools.combinations(qcs, 2)):
        if cpnet_simple.has_edge(qc1, qc2):
            graph.add_edge(qc1, qc2, weight=1.0)

    for ac1, ac2 in list(itertools.combinations(acs, 2)):
        if cpnet_simple.has_edge(ac1, ac2):
            graph.add_edge(ac1, ac2, weight=1.0)

    if len(qcs) == 0:
        qcs.append(-1)

    if len(acs) == 0:
        acs.append(-1)

    if len(paths) == 0:
        for qc in qcs:
            for ac in acs:
                graph.add_edge(qc,ac, rel=-1, weight=0.1)

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid') # re-index
    g_str = json.dumps(nx.node_link_data(g))
    return g_str


# relational graph generation
def relational_graph_generation(qcs, acs, paths, rels):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
    # print("qcs", qcs)
    # print("acs", acs)
    # print("paths", paths)
    # print("rels", rels)

    graph = nx.MultiDiGraph()
    for index, p in enumerate(paths):
        rel_list = rels[index]
        for c_index in range(len(p)-1):
            h = p[c_index]
            t = p[c_index+1]
            if graph.has_edge(h,t):
                existing_r_set = set([graph[h][t][r]["rel"] for r in graph[h][t]])
            else:
                existing_r_set = set()
            for r in rel_list[c_index]:
                # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
                # TODO: do we need to add both directions?
                if r in existing_r_set:
                    continue
                graph.add_edge(h,t, rel=r, weight=1.0)

    for qc1, qc2 in list(itertools.combinations(qcs, 2)):
        if cpnet_simple.has_edge(qc1, qc2):
            rs = get_edge(qc1, qc2)
            for r in rs:
                graph.add_edge(qc1, qc2, rel=r, weight=1.0)

    for ac1, ac2 in list(itertools.combinations(acs, 2)):
        if cpnet_simple.has_edge(ac1, ac2):
            rs = get_edge(ac1, ac2)
            for r in rs:
                graph.add_edge(ac1, ac2, rel=r, weight=1.0)

    if len(qcs) == 0:
        qcs.append(-1)

    if len(acs) == 0:
        acs.append(-1)

    if len(paths) == 0:
        for qc in qcs:
            for ac in acs:
                graph.add_edge(qc,ac, rel=-1, weight=0.1)

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid') # re-index
    g_str = json.dumps(nx.node_link_data(g))
    return g_str

def main():
    global pf_data, mcp_data
    global cpnet, concept2id, relation2id, id2relation, id2concept
    load_cpnet()
    load_resources()
    final_text = ""
    for index, qa_pairs in tqdm(enumerate(pf_data), desc="Building Graphs", total=len(pf_data)):
        # print(mcp_data[index])
        # print(pf_data[index])
        # print(qa_pairs)
        statement_paths = []
        statement_rel_list = []
        for qa_idx, qas in enumerate(qa_pairs):
            if qas["pf_res"] is None:
                cur_paths = []
                cur_rels = []
            else:
                cur_paths = [item["path"] for item in qas["pf_res"]]
                cur_rels = [item["rel"] for item in qas["pf_res"]]
            statement_paths.extend(cur_paths)
            statement_rel_list.extend(cur_rels)

        qcs = [concept2id[c] for c in mcp_data[index]["qc"]]
        acs = [concept2id[c] for c in mcp_data[index]["ac"]]

        gstr = plain_graph_generation(qcs=qcs, acs=acs,
                            paths=statement_paths,
                         rels=statement_rel_list)
        final_text += gstr + "\n"
    with open(GRAPH_PATH, 'w') as fw:
        fw.write(final_text)
    print("Write Graph Done: %s"%GRAPH_PATH)

main()