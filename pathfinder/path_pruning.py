import pickle
from tqdm import tqdm

import sys
flag = sys.argv[1]
threshold = 0.15
ori_pckle_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.pickle"%flag
scores_pckle_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.cls.scores.pickle"%flag
pruned_pckle_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.cls.pruned.%s.pickle"%(flag, str(threshold))

# threshold = 0.75
# threshold = 0.15


ori_paths = []
with open(ori_pckle_file, "rb") as fi:
    ori_paths = pickle.load(fi)

all_scores = []
with open(scores_pckle_file, "rb") as fi:
    all_scores = pickle.load(fi)


assert len(ori_paths) == len(all_scores)

ori_len = 0
pruned_len = 0
for index, qa_pairs in tqdm(enumerate(ori_paths[:]), desc="Scoring the paths", total=len(ori_paths)):
    for qa_idx, qas in enumerate(qa_pairs):
        statement_paths = qas["pf_res"]
        if statement_paths is not None:
            pruned_statement_paths = []
            for pf_idx, item in enumerate(statement_paths):
                score = all_scores[index][qa_idx][pf_idx]
                if score >= threshold:
                    pruned_statement_paths.append(item)
            ori_len += len(ori_paths[index][qa_idx]["pf_res"])
            pruned_len += len(pruned_statement_paths)
            assert len(ori_paths[index][qa_idx]["pf_res"]) >= len(pruned_statement_paths)
            ori_paths[index][qa_idx]["pf_res"] = pruned_statement_paths

print("ori_len:", ori_len, "\t\tafter_pruned_len:", pruned_len, "keep rate: %.4f"%(pruned_len/ori_len))
print("saving the pruned paths")
with open(pruned_pckle_file, 'wb') as fp:
    pickle.dump(ori_paths, fp, protocol=pickle.HIGHEST_PROTOCOL)
print("done!")