from OpenIE import get_relation_templates
import bisect
import configparser
import itertools
import operator


def collect_all_templates(sample_size=50):
    config = configparser.ConfigParser()
    config.read("paths.cfg")

    with open(config["paths"]["downsampled_relations"], "r", encoding="utf8") as f:
        downsampled_relations = set()
        for line in f.readlines():
            # Strip whitespace to remove trailing newline.
            downsampled_relations.add(line.strip())

    with open(config["paths"]["conceptnet_en"], "r", encoding="utf8") as f:
        relation_dict = {}

        for line in f.readlines():
            ls = line.strip().split('\t')
            rel = ls[0]
            head = ls[1]
            tail = ls[2]
            weight = float(ls[3])

            if rel not in downsampled_relations:
                continue

            if rel not in relation_dict:
                relation_dict[rel] = []
            relation_dict[rel].append((head, tail, weight))

    to_write = []
    for rel, rel_list in relation_dict.items():
        print(rel.upper(), flush=True)

        rel_list.sort(key=operator.itemgetter(2))
        rel_list_max = rel_list[-sample_size:]

        concept_pairs = [(head, tail) for head, tail, rel in rel_list_max]

        templates_dict = get_relation_templates(concept_pairs)
        
        templates_sorted = sorted(templates_dict.items(), key=operator.itemgetter(1), reverse=True)
        for template, freq in templates_sorted:
            to_write.append("\t".join([rel, template, str(freq)]))

    with open(config["paths"]["relation_templates_all"], "w", encoding="utf8") as f:
        f.write("\n".join(to_write))


if __name__ == "__main__":
    collect_all_templates(50000)
