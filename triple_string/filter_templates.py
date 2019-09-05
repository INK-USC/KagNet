import configparser
from nltk.corpus import stopwords as nltk_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
nltk_stop_words_set = set(nltk_stop_words.words("english"))
spacy_stop_words_set = set(spacy_stop_words)
def filter_stop_words(words):
    """
    Removes stop words from a given list of strings by comparing against spacy and nltk default stop words. Maintains
    original capitalization.
    :param words: List of strings to filter.
    :return: List of strings.
    """
    filtered = {w for w in words if w.lower() not in spacy_stop_words_set and w.lower() not in nltk_stop_words_set}
    return filtered

def read_all_templates(path):
    """
    Reads template file produced by collect_all_templates.py.
    """
    with open(path, encoding="utf8") as f:
        relation_translations = {}
        for line in f.readlines():
            ls = line.strip().split("\t")
            rel = ls[0]
            trans = ls[1]
            freq = ls[2] 
            if " </span> " in trans:
                continue
            trans = trans.replace("'s ", "is ")
            if rel not in relation_translations:
                relation_translations[rel] = []
            relation_translations[rel].append((rel, trans.lower(), freq))
        return relation_translations
 


def filter_templates():
    config = configparser.ConfigParser()
    config.read("paths.cfg")

    relation_translations = read_all_templates(config["paths"]["relation_templates_all"])

    # Build dictionary of each translation to how many times it appears across ALL relations, so we can check if a
    # translation is unique.
    trans_to_freq = {}
    for rel_trans_list in relation_translations.values():
        for rel, trans, freq in rel_trans_list:
            if trans not in trans_to_freq:
                trans_to_freq[trans] = 0
            trans_to_freq[trans] += 1

    # Filtering.
    relation_translations_filtered = {}
    for rel_trans_list in relation_translations.values():
        for rel, trans, freq in rel_trans_list:
            # Only collect translations that are unique and consist of multiple words.
            is_unique_trans = (trans_to_freq[trans] <= 2)
            is_multi_word = (len(filter_stop_words(trans.strip().split(" "))) >= 1 and len(trans.strip().split(" "))>=2)
            if is_unique_trans and is_multi_word:
                if rel not in relation_translations_filtered:
                    relation_translations_filtered[rel] = []
                relation_translations_filtered[rel].append((rel, trans, freq))

    to_write = []
    for rel_trans_list in relation_translations_filtered.values():
        # Save the top 15 translations for each relation.
        top = rel_trans_list[:50]
        top_as_str = ["\t".join([rel, "#SUBJ# %s #OBJ#"%trans, freq]) for rel, trans, freq in top]
        to_write.extend(top_as_str)

    with open(config["paths"]["relation_templates_filtered"], "w", encoding="utf8") as f:
        f.write("\n".join(to_write))


if __name__ == "__main__":
    filter_templates()
