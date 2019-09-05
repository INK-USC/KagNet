import operator
from bert_serving.client import BertClient
import configparser
import json
import numpy as np


def create_2_freq():
    config = configparser.ConfigParser()
    config.read("paths.cfg")

    with open(config["paths"]["triple_string_cpnet_json"], "r", encoding="utf8") as f:
        triple_str_json = json.load(f)

    print("Read " + str(len(triple_str_json)) + " triple strings.")

    concept_2_freq = {}
    relation_2_freq = {}

    for data in triple_str_json:
        words = data["string"].strip().split(" ")

        subj_start = data["subj_start"]
        subj_end = data["subj_end"]
        obj_start = data["obj_start"]
        obj_end = data["obj_end"]

        subj = "_".join(words[subj_start:subj_end])
        obj = "_".join(words[obj_start:obj_end])
        rel = data["rel"]

        if subj not in concept_2_freq:
            concept_2_freq[subj] = 0
        concept_2_freq[subj] += 1

        if obj not in concept_2_freq:
            concept_2_freq[obj] = 0
        concept_2_freq[obj] += 1

        if rel not in relation_2_freq:
            relation_2_freq[rel] = 0
        relation_2_freq[rel] += 1

    with open(config["paths"]["concept_2_freq"], "w", encoding="utf8") as f:
        sorted_x = sorted(concept_2_freq.items(), key=operator.itemgetter(1), reverse=True)
        for w, fre in sorted_x:
            f.write("%s\t%d\n"%(w, fre))

    with open(config["paths"]["relation_2_freq"], "w", encoding="utf8") as f:
        sorted_x = sorted(relation_2_freq.items(), key=operator.itemgetter(1), reverse=True)
        for w, fre in sorted_x:
            f.write("%s\t%d\n"%(w, fre))


if __name__ == "__main__":
    create_2_freq()
