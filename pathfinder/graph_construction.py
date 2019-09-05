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
import nltk
import json
# print('NLTK Version: %s' % (nltk.__version__))
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "in", "up", "may", "wanter"]

config = configparser.ConfigParser()
config.read("paths.cfg")

cpnet = None
concept2id = None
relation2id = None
id2relation = None
id2concept = None
blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])

def load_resources():
    global concept2id, relation2id, id2relation, id2concept
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

def save_cpnet():
    global concept2id, relation2id, id2relation, id2concept, blacklist
    load_resources()
    graph = nx.MultiDiGraph()
    with open(config["paths"]["conceptnet_en"], "r", encoding="utf8") as f:
        lines = f.readlines()

        def not_save(cpt):
            if cpt in blacklist:
                return True
            for t in cpt.split("_"):
                if t in nltk_stopwords:
                    return True
            return False

        for line in tqdm(lines, desc="saving to graph"):
            ls = line.strip().split('\t')
            rel = relation2id[ls[0]]
            subj = concept2id[ls[1]]
            obj = concept2id[ls[2]]
            weight = float(ls[3])
            if id2relation[rel] == "hascontext":
                continue
            if not_save(ls[1]) or not_save(ls[2]):
                continue
            if id2relation[rel] == "relatedto" or id2relation[rel] == "antonym":
                weight -= 0.3
                # continue
            if subj == obj: # delete loops
                continue
            weight = 1+float(math.exp(1-weight))
            graph.add_edge(subj, obj, rel=rel, weight=weight)
            graph.add_edge(obj, subj, rel=rel+len(relation2id), weight=weight)


    nx.write_gpickle(graph, config["paths"]["conceptnet_en_graph"])
    # with open(config["paths"]["conceptnet_en_graph"], 'w') as f:
    #     f.write(json.dumps(nx.node_link_data(graph)))

save_cpnet()