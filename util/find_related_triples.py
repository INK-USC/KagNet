import spacy
import sys

PATH_relation2id = "relation2id.txt"
PATH_entity2id = "entity2id.txt"
PATH_triple2id = "triple2id.txt"
PATH_related_triples = "related_triples.txt"
PATH_related_triples_str = "related_triples_str.txt"

nlp = spacy.load('en')


def find_related_triples(words, k=3):
    """
    Given an input sentence, this function finds ConceptNet triplets (relationships) that are one hop away from entities
    present in the input sentence.
    :param words: Words in sentence, as list of strings.
    :param k: Max n-gram size to search up to for multi-word concepts.
    :return: Set of triple ids for triples related (1-hop) to the sentence.
    """

    # Filter out any strings that don't contain any letters, e.g. '.'
    words = [w for w in words if any(c.isalpha() for c in w)]

    # Generate all n-grams up to size k.
    concepts = set()
    for n in range(k):
        for i in range(len(words) - n):
            concepts.add(" ".join(words[i:i + n + 1:]))

    # Filter out stop words.
    concepts = {c for c in concepts if not nlp.vocab[c].is_stop}

    # Convert lowercase.
    concepts = {c.lower() for c in concepts}

    # Replace spaces with _ (this is how ConceptNet stores multi-word entities.
    concepts = {c.replace(" ", "_") for c in concepts}

    print("Potential concepts: " + str(concepts))

    # Read in mappings from files.
    entity_to_id, id_to_entity = read_entity2id()
    relation_to_id, id_to_relation = read_relation2id()

    # Convert concepts to their entity IDs.
    concept_ids = {entity_to_id[c] for c in concepts if c in entity_to_id}

    print("Identified concepts: " + str({id_to_entity[c] for c in concept_ids}))

    # Create triples dictionary.
    triples_dict = create_triples_dict()

    related_triples = set()
    for c_id in concept_ids:
        if c_id in triples_dict:
            related_triples.update(triples_dict[c_id])

    print("Found " + str(len(related_triples)) + " related triples.")

    with open(PATH_related_triples, 'w', encoding="utf8") as fout:
        fout.write(str(len(related_triples)) + "\n")
        fout.write(" ".join(related_triples))

    return related_triples


def read_entity2id():
    """
    Reads entity-to-id mapping from file.
    :return: Tuple containing entity-to-id and id-to-entity map.
    """
    entity_to_id = {}
    id_to_entity = {}
    with open(PATH_entity2id, 'r', encoding="utf8") as f:
        for line in f.readlines():
            elements = line.strip().split()
            if len(elements) < 2:
                continue
            entity_str = elements[0].split("/")[-1]
            entity_id = elements[1]
            entity_to_id[entity_str] = entity_id
            id_to_entity[entity_id] = entity_str
    return entity_to_id, id_to_entity


def read_relation2id():
    """
    Reads relation-to-id mapping from file.
    :return: Tuple containing relation-to-id and id-to-relation map.
    """
    relation_to_id = {}
    id_to_relation = {}
    with open(PATH_relation2id, 'r', encoding="utf8") as f:
        for line in f.readlines():
            elements = line.strip().split()
            if len(elements) < 2:
                continue
            relation_str = elements[0].split("/")[-1]
            relation_id = elements[1]
            relation_to_id[relation_str] = relation_id
            id_to_relation[relation_id] = relation_str
    return relation_to_id, id_to_relation


def read_triple2id():
    """
    Reads triple-to-id mapping from file.
    :return: Tuple containing triple-to-id and id-to-triple map. Triple is represented as tuple of (h, t, r), and triple
        id is a number.
    """
    triple_to_id = {}
    id_to_triple = {}
    with open(PATH_triple2id, 'r', encoding="utf8") as f:
        for line in f.readlines():
            elements = line.strip().split()
            if len(elements) < 4:
                continue

            triple_id = elements[0]
            head_id = elements[1]
            tail_id = elements[2]
            relation_id = elements[3]

            triple_to_id[(head_id, tail_id, relation_id)] = triple_id
            id_to_triple[triple_id] = (head_id, tail_id, relation_id)
    return triple_to_id, id_to_triple


def create_triples_dict():
    triples_dict = {}
    triple_to_id, id_to_triple = read_triple2id()
    for triple, triple_id in triple_to_id.items():
        head_id = triple[0]
        tail_id = triple[1]

        if head_id not in triples_dict:
            triples_dict[head_id] = []
        if tail_id not in triples_dict:
            triples_dict[tail_id] = []

        triples_dict[head_id].append(triple_id)
        triples_dict[tail_id].append(triple_id)

    return triples_dict


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_related_triples.py <sentence>")
        sys.exit()

    sentence = sys.argv[1]
    doc = nlp(sentence)
    sentence_words = [token.text for token in doc]

    find_related_triples(sentence_words)

