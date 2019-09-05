
# to run on server
import matplotlib
matplotlib.use('Agg')

import plotly
import plotly.plotly as py
import plotly.tools as tls
import pickle



import matplotlib.pyplot as plt

import numpy as np



all_scores = []
pckle_filename = "../datasets/csqa_new/dev_rand_split.jsonl.statements.mcp.pf.cls.scores.pickle"
with open(pckle_filename, "rb") as fi:
    all_scores = pickle.load(fi)

scores = []
for statement in all_scores:
    for qa_pair_paths in statement:
        if qa_pair_paths is None:
            continue
        scores.extend(qa_pair_paths)

# plotly.tools.set_credentials_file(username='xxx', api_key='xxxxx')
# print(scores)
plt.hist(scores, bins=100)
plt.title("Path Score Hist (triple cls)")
plt.xlabel("Path Score")
plt.ylabel("Frequency")

fig = plt.gcf()
plotly_fig = tls.mpl_to_plotly(fig)
py.iplot(plotly_fig, filename='dev path score (triple cls)')



