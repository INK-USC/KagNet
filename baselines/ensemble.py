# output_files = []
# output_files.append("models/bert_large_b60g4lr1e-4wd0.01wp0.1_1337_dev_output.csv")
# output_files.append("models/bert_large_b60g4lr1e-4wd0.01wp0.1_42_dev_output.csv")
# output_files.append("models/bert_large_b60g4lr1e-4wd0.01wp0.1_73_dev_output.csv")
# output_files.append("models/bert_large_b60g4lr1e-4wd0.01wp0.1_233_dev_output.csv")
# output_files.append("models/bert_large_b60g4lr1e-4wd0.01wp0.1_1024_dev_output.csv")
#
# final_output_file = "models/bert_large_b60g4lr1e-4wd0.01wp0.1_final_dev_output.csv"

output_files = []
# output_files.append("models/bert_large_b60g4lr1e-4wd0.01wp0.1_1337wsc_prediction.csv ")
output_files.append("models/bert_large_b60g4lr1e-4wd0.01wp0.1_42wsc_prediction.csv")
# output_files.append("models/bert_large_b60g4lr1e-4wd0.01wp0.1_73wsc_prediction.csv ")
output_files.append("models/bert_large_b60g4lr1e-4wd0.01wp0.1_233wsc_prediction.csv")
output_files.append("models/bert_large_b60g4lr1e-4wd0.01wp0.1_1024wsc_prediction.csv")

final_output_file = "models/bert_large_b60g4lr1e-4wd0.01wp0.1_finalwsc_prediction.csv"


anss = {}
qids = []
agree_res = [0,0,0,0,0,0]

for output_file in output_files:
    cur_qids = []

    with open(output_file, 'r') as fp:
        for line in fp.readlines():
            ls = line.strip().split(",")
            qid = ls[0]
            ans = ls[1]
            cur_qids.append(qid)
            if qid not in anss:
                anss[qid] = []
            anss[qid].append(ans)

    if len(qids) == 0:
        qids = cur_qids
    else:
        assert cur_qids == qids

## output

def most_frequent(List, qid=0):
    res = max(sorted(list(set(List))), key=List.count)
    agree = 0
    for i in List:
        if i==res:
            agree +=1

    if agree == 2 or agree==5:
        print(qid+","+res)
    agree_res[agree] += 1

    return res
#     # return List[4]

# def most_frequent(lst):
#     return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]

final_ans = {}
for qid in qids:
    ans = anss[qid]
    f_ans = most_frequent(ans, qid=qid)
    final_ans[qid] = f_ans

with open(final_output_file, 'w') as fout:
    for qid in qids:
        fout.write(",".join((qid, final_ans[qid])) + "\n")

print(agree_res[1:])
print(len(final_ans))