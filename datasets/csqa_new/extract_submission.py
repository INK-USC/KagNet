import json
with open("test_rand_split.jsonl", 'r') as fw:
    for line in open("test_rand_split.jsonl", 'r').readlines():
        data = json.loads(line.strip())
        print(data["id"]+','+data["answerKey"])