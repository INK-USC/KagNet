from tqdm import tqdm
import nltk
import json
# print('NLTK Version: %s' % (nltk.__version__))
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "in", "up", "may", "wanter"]
print(nltk_stopwords)
# print(nltk_stopwords)
import configparser
import sys

data = []



concept_vocab = set()
config = configparser.ConfigParser()
config.read("paths.cfg")
with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
    cpnet_vocab = set([l.strip() for l in list(f.readlines())])

# cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]

# python prune_qc.py ../datasets/csqa_new/dev_rand_split.jsonl.statements.mcp
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

prune_data = []
for item in tqdm(data):
    qc = item["qc"]
    prune_qc = []
    for c in qc:
        if c[-2:] == "er" and c[:-2] in qc:
            continue
        if c[-1:] == "e" and c[:-1] in qc:
            continue
        have_stop = False
        for t in c.split("_"):
            if t in nltk_stopwords:
                have_stop = True
        if not have_stop and c in cpnet_vocab:
            prune_qc.append(c)

    ac = item["ac"]
    prune_ac = []
    for c in ac:
        if c[-2:] == "er" and c[:-2] in ac:
            continue
        if c[-1:] == "e" and c[:-1] in ac:
            continue
        all_stop = True
        for t in c.split("_"):
            if t not in nltk_stopwords:
                all_stop = False
        if not all_stop and c in cpnet_vocab:
            prune_ac.append(c)

    item["qc"] = prune_qc
    item["ac"] = prune_ac

    prune_data.append(item)

import jsbeautifier
opts = jsbeautifier.default_options()
opts.indent_size = 2

with open(sys.argv[1], 'w') as fp:
    # json.dump(prune_data, f)
    fp.write(jsbeautifier.beautify(json.dumps(prune_data), opts))