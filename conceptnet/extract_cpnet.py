import configparser
import json

relation_mapping = dict()


def load_merge_relation():
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    with open(config["paths"]["merge_relation"], encoding="utf8") as f:
        for line in f.readlines():
            ls = line.strip().split('/')
            rel = ls[0]
            for l in ls:
                if l.startswith("*"):
                    relation_mapping[l[1:]] = "*" + rel
                else:
                    relation_mapping[l] = rel


def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def extract_english():
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    config = configparser.ConfigParser()
    config.read("paths.cfg")

    only_english = []
    with open(config["paths"]["conceptnet"], encoding="utf8") as f:
        for line in f.readlines():
            ls = line.split('\t')
            if ls[2].startswith('/c/en/') and ls[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = ls[1].split("/")[-1].lower()
                head = del_pos(ls[2]).split("/")[-1].lower()
                tail = del_pos(ls[3]).split("/")[-1].lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue

                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue

                if rel not in relation_mapping:
                    continue
                rel = relation_mapping[rel]
                if rel.startswith("*"):
                    rel = rel[1:]
                    tmp = head
                    head = tail
                    tail = tmp

                data = json.loads(ls[4])

                only_english.append("\t".join([rel, head, tail, str(data["weight"])]))

    with open(config["paths"]["conceptnet_en"], "w", encoding="utf8") as f:
        f.write("\n".join(only_english))


if __name__ == "__main__":
    load_merge_relation()
    print(relation_mapping)
    extract_english()