import configparser
import json
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import nltk
# print('NLTK Version: %s' % (nltk.__version__))
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
# print(nltk_stopwords)

def create_pattern(nlp, doc):
    pronoun_list = set(["my", "you", "it", "its", "your","i","he", "she","his","her","they","them","their","our","we"])
    # Filtering concepts consisting of all stop words and longer than four words.
    if len(doc) >= 5 or doc[0].text in pronoun_list or doc[-1].text in pronoun_list or all([(token.text in nltk_stopwords or token.lemma_ in nltk_stopwords) for token in doc]):  #
        return None  # ignore this concept as pattern

    pattern = []
    for token in doc:  # a doc is a concept
        pattern.append({"LEMMA": token.lemma_})
    return pattern

def create_matcher_patterns():
    config = configparser.ConfigParser()
    config.read("paths.cfg")

    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        cpnet_vocab = [l.strip() for l in list(f.readlines())]


    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]


    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

    docs = nlp.pipe(cpnet_vocab)

    all_patterns = {}

    for doc in tqdm(docs, total=len(cpnet_vocab)):

        pattern = create_pattern(nlp, doc)

        if pattern is None:
            continue
        all_patterns["_".join(doc.text.split(" "))] = pattern

    print("Created " + str(len(all_patterns)) + " patterns.")

    with open(config["paths"]["matcher_patterns"], "w", encoding="utf8") as f:
        json.dump(all_patterns, f)


if __name__ == "__main__":
    create_matcher_patterns()
