import configparser
import json

def get_triple_string_sentences():
    config = configparser.ConfigParser()
    config.read("paths.cfg")

    with open(config["paths"]["tp_str_corpus"], "r", encoding="utf8") as f:
        triple_str_json = json.load(f)

    print("Read " + str(len(triple_str_json)) + " triple strings.")

    to_write = []
    for data in triple_str_json:
        to_write.append(data["string"])

    with open(config["paths"]["tp_str_sentences"], "w", encoding="utf8") as f:
        f.write("\n".join(to_write))


if __name__ == "__main__":
    get_triple_string_sentences()