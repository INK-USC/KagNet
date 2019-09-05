

import gzip

import numpy as np

from collections import OrderedDict

import pickle

from tqdm import tqdm

import mmap

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


# with open("numberbatch-en-17.06.txt", "r") as f:
#     _ = f.readline()
#     lines = f.readlines()
#     i = 0
#     for line in lines:
#         i+=1
#         if i >=100:
#             break
#         line= line.split(' ')
#
#         if (line[0].startswith('/c/en/')):
#             print(line[0])
         # ls[0].startswith('/c/en/') and ls[3].startswith('/c/en/'):
         #    """
         #    Some preprocessing:
         #        - Remove part-of-speech encoding.
         #        - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to
         #        - Lowercase for uniformity.
         #    """
         #    rel = ls[1].split("/")[-1].lower()
         #    head = del_pos(ls[2]).split("/")[-1].lower()
         #    tail = del_pos(ls[3]).split("/")[-1].lower()



'''
In numberbatch, concepts whose name include "acquire":
acquire_knowledge
acquired_hemochromatosis                           
acquired_knowledge                                                                     
acquiree                                                                                                   
acquirer
....
but numberbatch doesn't have the concept "acquire"
'''

concept_dic = {}

concepts = []
with open("concept.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines[:200]:
        line = line.strip()
        concepts.append(line)

concept_dic = concept_dic.fromkeys(concepts)

dic_size = len(concepts)
print(dic_size)


with open("numberbatch-17.06.txt", "r", encoding="utf8") as f:


    # line = f.readline()


    for line in tqdm(f, total=get_num_lines("numberbatch-17.06.txt")):

        assert len(concept_dic) == dic_size

        line = f.readline().strip()

        line= line.split(' ')

        if (line[0].startswith('/c/en/')):
            concept = del_pos(line[0]).split("/")[-1].lower()
            if "acquire" in concept:
                print(concept)

            if not concept.replace("_", "").replace("-", "").isalpha():
                continue

            if concept in concept_dic.keys():
                concept_dic[concept] = np.array(line[1:]).astype(np.float32)

exit()
concept_dic = OrderedDict(sorted(concept_dic.items(), key=lambda t: concepts.index(t[0])))


f = open('concept_dic.dat', 'wb')
try:
    pickle.dump(concept_dic, f)
except Exception as e:
    print(e)

keys = list(concept_dic.keys())
values = list(concept_dic.values())

print(keys[:10])
print(values[:10])

#'ab_extra', 'ab_intra', 'actinal', 'abandon', 'acquire', 'arrogate'