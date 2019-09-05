import argparse
import csv
import json
import sys
from tqdm import tqdm


def add_answer_to_ccr_csqa(ccr_path, dataset_path, output_path):
    with open(ccr_path, "r", encoding="utf8") as f:
        ccr = json.load(f)

    with open(dataset_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f.readlines()):
            csqa_json = json.loads(line)
            for j, ans in enumerate(csqa_json["question"]["choices"]):
                ccr[i * 5 + j]["stem"] = csqa_json["question"]["stem"].lower()
                ccr[i * 5 + j]["answer"] = ans["text"].lower()

    with open(output_path, "w", encoding="utf8") as f:
        import jsbeautifier
        opts = jsbeautifier.default_options()
        opts.indent_size = 2
        f.write(jsbeautifier.beautify(json.dumps(ccr), opts))


def add_answer_to_ccr_swag(ccr_path, dataset_path, output_path):
    with open(ccr_path, "r", encoding="utf8") as f:
        ccr = json.load(f)

    with open(dataset_path, "r", encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip first line (header).
        for i, row in tqdm(enumerate(reader)):
            stem = row[4] + " " + row[5]
            answers = row[7:11]

            for j, ans in enumerate(answers):
                ccr[i * 4 + j]["stem"] = stem.lower()
                ccr[i * 4 + j]["answer"] = ans.lower()

    with open(output_path, "w", encoding="utf8") as f:
        import jsbeautifier
        opts = jsbeautifier.default_options()
        opts.indent_size = 2
        f.write(jsbeautifier.beautify(json.dumps(ccr), opts))
        #json.dump(ccr, f)


# python add_answer_to_ccr.py --mode csqa --ccr_path csqa_new/train_rand_split.jsonl.statements.ccr --dataset_path csqa_new/train_rand_split.jsonl.statements --output_path csqa_new/train_rand_split.jsonl.statements.ccr.a
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest="mode")
    parser.add_argument("--ccr_path", dest="ccr_path")
    parser.add_argument("--dataset_path", dest="dataset_path")
    parser.add_argument("--output_path", dest="output_path")

    args = parser.parse_args()

    if args.mode == "swag":
        add_answer_to_ccr_swag(args.ccr_path, args.dataset_path, args.output_path)
    elif args.mode == "csqa":
        add_answer_to_ccr_csqa(args.ccr_path, args.dataset_path, args.output_path)
