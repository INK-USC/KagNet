import configparser
import json
import random

random.seed(42)
template = {}
config = configparser.ConfigParser()
config.read("paths.cfg")

def load_templates():
    with open(config["paths"]["tp_str_template"], encoding="utf8") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("["):
                rel = line.split('/')[0][1:]
                if rel.endswith(']'):
                    rel = rel[:-1]
                template[rel] = []
            elif "#SUBJ#" in line and  "#OBJ#" in line:
                template[rel].append(line)

def generate_triple_string(tid, subj, rel, obj):
    temp = random.choice(template[rel])
    tp_str = {"tid": tid}
    tp_str["rel"] = rel
    tp_str["subj"] = subj
    tp_str["obj"] = obj
    tp_str["temp"] = temp

    subj = subj.replace('_', ' ')
    obj = obj.replace('_', ' ')
    tp_str["string"] = temp.replace("#SUBJ#", subj).replace("#OBJ#", obj)

    subj_lst = subj.split()
    obj_lst = obj.split()
    tp_str_lst = tp_str["string"].split()
    tp_str["subj_start"] = 0

    # print(subj, rel, obj, tp_str["string"])

    while tp_str_lst[tp_str["subj_start"]: tp_str["subj_start"]+len(subj_lst)] != subj_lst:
        tp_str["subj_start"] += 1
    tp_str["subj_end"] = tp_str["subj_start"] + len(subj_lst)

    tp_str["obj_start"] = 0
    while tp_str_lst[tp_str["obj_start"]: tp_str["obj_start"]+len(obj_lst)] != obj_lst:
        tp_str["obj_start"] += 1
    tp_str["obj_end"] = tp_str["obj_start"] + len(obj_lst)
    return tp_str

def create_corpus():
    corpus = []
    with open(config["paths"]["conceptnet_en"], "r", encoding="utf8") as f:
        for line in f.readlines():
            ls = line.strip().split('\t')
            rel = ls[0]
            head = ls[1]
            tail = ls[2]
            tp_str = generate_triple_string(len(corpus), head, rel, tail)
            corpus.append(tp_str)
    with open(config["paths"]["tp_str_corpus"], "w", encoding="utf8") as f:
        print("Writing Json File....")
        json.dump(corpus, f)

if __name__ == "__main__":
    load_templates()
    # print(template)
    # tp_str = generate_triple_string(1,"love","antonym","hate")
    create_corpus()
