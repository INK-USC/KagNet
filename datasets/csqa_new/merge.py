'''
test_no_answer_file = "test_rand_split_no_answers.jsonl"
submission_file = "sgn-lite.csv.submitted"
merged_file = "test_rand_split.jsonl.backup"
predictions = [line.strip().split(",") for line in open(submission_file, 'r').readlines()]

import json
index = 0
with open(merged_file, 'w') as fw:
    for line in open(test_no_answer_file, 'r').readlines():
        data = json.loads(line.strip())
        assert predictions[index][0] == data["id"]
        data["answerKey"] = predictions[index][1]
        index += 1
        fw.write(json.dumps(data)+"\n")
'''





filterd_id_file = "trust_test.id.txt"

qids = [line.strip().split(",")[0] for line in open(filterd_id_file, 'r').readlines()]


import json
with open("train2_rand_split.jsonl", 'w') as fw:
    for line in open("test_rand_split.jsonl", 'r').readlines():
        data = json.loads(line.strip())
        if data["id"] not in qids:
            continue
        fw.write(json.dumps(data)+"\n")







