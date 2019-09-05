import json
import configparser
import numpy as np


'''
input file: direct output of TransE (json file)

output file: two npy files storing the embedding matrix of entities and relations respectively,
             the file does not deal with the names of entities and relations.
'''

config = configparser.ConfigParser()
config.read("paths.cfg")

transe_res = config["paths"]["transe_res"]


'''remove ".vec.json" from filename'''
output_name = ".".join(transe_res.split('.')[: -2]).lower()
ent_embeddings_file = output_name + '.ent.npy'
rel_embeddings_file = output_name + '.rel.npy'


with open(transe_res, "r") as f:
    dic = json.load(f)

ent_embs, rel_embs = dic['ent_embeddings'], dic['rel_embeddings']

ent_embs = np.array(ent_embs, dtype="float32")
rel_embs = np.array(rel_embs, dtype="float32")

np.save(ent_embeddings_file, ent_embs)
np.save(rel_embeddings_file, rel_embs)


