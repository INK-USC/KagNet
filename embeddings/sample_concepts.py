import configparser
import json
import numpy as np


def sample_concepts(sample_size=1000):
    config = configparser.ConfigParser()
    config.read("paths.cfg")

    with open(config["paths"]["concept_2_freq"], "r", encoding="utf8") as f:
        concept_2_freq = json.load(f)

    concepts = list(concept_2_freq.keys())
    weights = list(concept_2_freq.values())
    weights = [float(w) for w in weights]
    total = sum(weights)
    weights = [w / total for w in weights]

    sample = np.random.choice(concepts, size=sample_size, p=weights)

    sample = [s.replace(" ", "_") for s in sample]

    with open("concept_samples.txt", "w", encoding="utf8") as f:
        f.write("\n".join(sample))


if __name__ == "__main__":
    sample_concepts()
