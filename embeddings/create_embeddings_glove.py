import configparser
import json
import numpy as np
import sys
from tqdm import tqdm


def load_glove_from_npy(glove_vec_path, glove_vocab_path):
    vectors = np.load(glove_vec_path)
    with open(glove_vocab_path, "r", encoding="utf8") as f:
        vocab = [l.strip() for l in f.readlines()]

    assert(len(vectors) == len(vocab))

    glove_embeddings = {}
    for i in range(0, len(vectors)):
        glove_embeddings[vocab[i]] = vectors[i]
    print("Read " + str(len(glove_embeddings)) + " glove vectors.")
    return glove_embeddings


def weighted_average(avg, new, n):
    # TODO: maybe a better name for this function?
    return ((n - 1) / n) * avg + (new / n)


def max_pooling(old, new):
    # TODO: maybe a better name for this function?
    return np.maximum(old, new)


def write_embeddings_npy(embeddings, embeddings_cnt, npy_path, vocab_path):
    words = []
    vectors = []
    for key, vec in embeddings.items():
        words.append(key)
        vectors.append(vec)

    matrix = np.array(vectors, dtype="float32")
    print(matrix.shape)

    print("Writing embeddings matrix to " + npy_path, flush=True)
    np.save(npy_path, matrix)
    print("Finished writing embeddings matrix to " + npy_path, flush=True)

    print("Writing vocab file to " + vocab_path, flush=True)
    to_write = ["\t".join([w, str(embeddings_cnt[w])]) for w in words]
    with open(vocab_path, "w", encoding="utf8") as f:
        f.write("\n".join(to_write))
    print("Finished writing vocab file to " + vocab_path, flush=True)


def create_embeddings_glove(pooling="max", dim=100):
    print("Pooling: " + pooling)

    config = configparser.ConfigParser()
    config.read("paths.cfg")

    with open(config["paths"]["triple_string_cpnet_json"], "r", encoding="utf8") as f:
        triple_str_json = json.load(f)
    print("Loaded " + str(len(triple_str_json)) + " triple strings.")

    glove_embeddings = load_glove_from_npy(config["paths"]["glove_vec_npy"], config["paths"]["glove_vocab"])
    print("Loaded glove.", flush=True)

    concept_embeddings = {}
    concept_embeddings_cnt = {}
    rel_embeddings = {}
    rel_embeddings_cnt = {}

    for i in tqdm(range(len(triple_str_json))):
        data = triple_str_json[i]

        words = data["string"].strip().split(" ")

        rel = data["rel"]
        subj_start = data["subj_start"]
        subj_end = data["subj_end"]
        obj_start = data["obj_start"]
        obj_end = data["obj_end"]

        subj_words = words[subj_start:subj_end]
        obj_words = words[obj_start:obj_end]

        subj = " ".join(subj_words)
        obj = " ".join(obj_words)

        # counting the frequency (only used for the avg pooling)
        if subj not in concept_embeddings:
            concept_embeddings[subj] = np.zeros((dim,))
            concept_embeddings_cnt[subj] = 0
        concept_embeddings_cnt[subj] += 1

        if obj not in concept_embeddings:
            concept_embeddings[obj] = np.zeros((dim,))
            concept_embeddings_cnt[obj] = 0
        concept_embeddings_cnt[obj] += 1

        if rel not in rel_embeddings:
            rel_embeddings[rel] = np.zeros((dim,))
            rel_embeddings_cnt[rel] = 0
        rel_embeddings_cnt[rel] += 1

        if pooling == "avg":
            subj_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in subj_words])
            obj_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in obj_words])

            if rel in ["relatedto", "antonym"]:
                # Symmetric relation.
                rel_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in words]) - subj_encoding_sum - obj_encoding_sum
            else:
                # Asymmetrical relation.
                rel_encoding_sum = obj_encoding_sum - subj_encoding_sum

            subj_len = subj_end - subj_start
            obj_len = obj_end - obj_start

            subj_encoding = subj_encoding_sum / subj_len
            obj_encoding = obj_encoding_sum / obj_len
            rel_encoding = rel_encoding_sum / (len(words) - subj_len - obj_len)

            concept_embeddings[subj] = subj_encoding
            concept_embeddings[obj] = obj_encoding
            rel_embeddings[rel] = weighted_average(rel_embeddings[rel], rel_encoding, rel_embeddings_cnt[rel])

        elif pooling == "max":
            subj_encoding = np.amax([glove_embeddings.get(word, np.zeros((dim,))) for word in subj_words], axis=0)
            obj_encoding = np.amax([glove_embeddings.get(word, np.zeros((dim,))) for word in obj_words], axis=0)

            mask_rel = []
            for j in range(len(words)):
                if subj_start <= j < subj_end or obj_start <= j < obj_end:
                    continue
                mask_rel.append(j)
            rel_vecs = [glove_embeddings.get(words[i], np.zeros((dim,))) for i in mask_rel]
            rel_encoding = np.amax(rel_vecs, axis=0)

            # here it is actually avg over max for relation
            concept_embeddings[subj] = max_pooling(concept_embeddings[subj], subj_encoding)
            concept_embeddings[obj] = max_pooling(concept_embeddings[obj], obj_encoding)
            rel_embeddings[rel] = weighted_average(rel_embeddings[rel], rel_encoding, rel_embeddings_cnt[rel])

    print(str(len(concept_embeddings)) + " concept embeddings")
    print(str(len(rel_embeddings)) + " relation embeddings")

    write_embeddings_npy(concept_embeddings, concept_embeddings_cnt,config["paths"]["concept_vec_npy_glove"] + "." + pooling,
                         config["paths"]["concept_vocab_glove"] + "." + pooling + ".txt")
    write_embeddings_npy(rel_embeddings, rel_embeddings_cnt, config["paths"]["relation_vec_npy_glove"] + "." + pooling,
                         config["paths"]["relation_vocab_glove"] + "." + pooling + ".txt")


if __name__ == "__main__":
    create_embeddings_glove()
