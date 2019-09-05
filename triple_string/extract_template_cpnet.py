import configparser
import json
import collections
import operator

rel_templates_dict = {}

def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def extract_tempalte():
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

                data = json.loads(ls[4])
                if "surfaceText" in data:
                    subj = data["surfaceStart"].lower()
                    obj = data["surfaceEnd"].lower()
                    surface = data["surfaceText"].lower()
                    temp = surface.replace("[[%s]]"%subj,"#SUBJ#").replace("[[%s]]"%obj,"#OBJ#")
                    temp = temp.replace('[','').replace(']','')
                    if '#SUBJ#' not in temp or '#OBJ#' not in temp:
                        continue
                    weight = data["weight"] 
                    if rel not in rel_templates_dict:
                        rel_templates_dict[rel] = dict()
                    if temp not in rel_templates_dict[rel]:
                        rel_templates_dict[rel][temp] = 0.0
                    rel_templates_dict[rel][temp] += weight                 



    with open(config["paths"]["conceptnet_en"], "w", encoding="utf8") as f:
        f.write("\n".join(only_english))


if __name__ == "__main__":
    extract_tempalte()
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    to_write = []
    print("#relation", len(rel_templates_dict))
    print(rel_templates_dict.keys())
    for rel in rel_templates_dict:
        templates_sorted = sorted(rel_templates_dict[rel].items(), key=operator.itemgetter(1), reverse=True)
        for template, freq in templates_sorted:
            to_write.append("\t".join([rel, template, str(freq)]))
    with open(config["paths"]["relation_templates_filtered_cpnet"], "w", encoding="utf8") as f:
        f.write("\n".join(to_write))
