PATH_triple2id = "triple2id.txt"


def append_triple_id():
    """
    Appends the triple ID to the start of each triple in the triple2id text file. Writes back to the same file.
    """

    triple_id = 0
    output = []

    with open(PATH_triple2id, "r", encoding="utf8") as f:
        for line in f.readlines():
            num_elements = len(line.strip().split())
            if num_elements != 3:
                continue
            output.append("\t".join([str(triple_id), line]))
            triple_id += 1

    with open(PATH_triple2id, "w", encoding="utf8") as fout:
        for line in output:
            fout.write(line)


if __name__ == "__main__":
    append_triple_id()
