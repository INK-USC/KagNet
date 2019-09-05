import configparser

config = configparser.ConfigParser()
config.read("paths.cfg")



def save_data():
    output_folder = config["paths"]["openie_fodler"]


    # write entity2id.txt
    entity2id = {}
    with open(config["paths"]["%s_vocab" % "concept"], "r", encoding="utf8") as f:
        vocab = [l.strip() for l in f.readlines()]
    for w in vocab:
        entity2id[w] = len(entity2id)

    with open(output_folder + "entity2id.txt", "w", encoding="utf8") as f:
        f.write("%d\n"%len(entity2id))
        for e in entity2id:
            f.write("%s\t%d\n" % (e, entity2id[e]))

    # write relation2id.txt
    relation2id = {}
    with open(config["paths"]["%s_vocab" % "relation"], "r", encoding="utf8") as f:
        vocab = [l.strip() for l in f.readlines()]
    for w in vocab:
        relation2id[w] = len(relation2id)

    with open(output_folder + "relation2id.txt", "w", encoding="utf8") as f:
        f.write("%d\n" % len(relation2id))
        for e in relation2id:
            f.write("%s\t%d\n" % (e, relation2id[e]))


    # write train2id.txt
    triples = []
    with open(config["paths"]["conceptnet_en"], "r", encoding="utf8") as f:
        for line in f.readlines():
            ls = line.strip().split('\t')
            rel = ls[0]
            subj = ls[1]
            obj = ls[2]
            assert subj in entity2id and obj in entity2id and rel in relation2id
            triples.append((entity2id[subj], relation2id[rel], entity2id[obj]))
    with open(output_folder + "train2id.txt", "w", encoding="utf8") as fw:
        fw.write("%d\n" % len(triples))
        for t in triples:
            fw.write("%d\t%d\t%d\n" % (t[0], t[2], t[1]))

    return triples

save_data()
