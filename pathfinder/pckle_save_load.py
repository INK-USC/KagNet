import timeit
import json
import pickle

fname = '../datasets/csqa_new/train_rand_split.jsonl.statements.mcp.pf'
#
start_time = timeit.default_timer()
print("loading paths from %s" % fname)
with open(fname, 'r') as f:
    pf_json_data = json.load(f)
print('\t Load Json Done! Time: ', "{0:.2f} sec".format(float(timeit.default_timer() - start_time)))


start_time = timeit.default_timer()
with open(fname+'.pickle', 'wb') as handle:
    pickle.dump(pf_json_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('\t Save Pickle Done! Time: ', "{0:.2f} sec".format(float(timeit.default_timer() - start_time)))

# start_time = timeit.default_timer()
# with open(fname+'.pickle', 'rb') as handle:
#     pf_json_data = pickle.load(handle)
# print('\t Load Pickle Done! Time: ', "{0:.2f} sec".format(float(timeit.default_timer() - start_time)))