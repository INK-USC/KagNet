import pickle
from tqdm import tqdm
import json
import timeit
import statistics

threshold = 0.17
PF_PATH = "../datasets/csqa_new/dev_rand_split.jsonl.statements.mcp.pf.cls.pruned.%s.pickle" % (str(threshold))
statement_json_file = "../datasets/csqa_new/dev_rand_split.jsonl.statements"
flag = "pruned"


def pathfinding_analysis(pf_pckle_file, statement_json_file, flag="original"):
    statement_json_data = []
    print("loading statements from %s" % statement_json_file)
    with open(statement_json_file, "r") as fp:
        for line in fp.readlines():
            statement_data = json.loads(line.strip())
            statement_json_data.append(statement_data)

    labels = []
    for question_id in range(len(statement_json_data)):
        for k, s in enumerate(statement_json_data[question_id]["statements"]):
            labels.append(s['label'])

    print("Done!")

    print("loading paths from %s" % pf_pckle_file)
    start_time = timeit.default_timer()
    path_json_data = []
    with open(pf_pckle_file, "rb") as fi:
        path_json_data = pickle.load(fi)
    print('\t Done! Time: ', "{0:.2f} sec".format(float(timeit.default_timer() - start_time)))

    assert len(path_json_data) == len(labels)

    correct_avg_path_lens_perQA = []
    wrong_avg_path_lens_perQA = []

    correct_path_counts = []
    wrong_path_counts = []

    correct_none_qa_pair_coverage = []
    wrong_none_qa_pair_coverage = []

    no_qa_pair = [0, 0]

    for index, qa_pairs in tqdm(enumerate(path_json_data[:]), desc="Scoring the paths", total=len(path_json_data)):
        qa_path_lenths = []
        coverd_qa_pair = 0
        path_counts = []
        for qa_idx, qas in enumerate(qa_pairs):
            if qas["pf_res"] is None:
                statement_paths = []
            else:
                statement_paths = [item["path"] for item in qas["pf_res"]]

            if len(statement_paths) == 0:
                qa_path_lenths.append(0)
            else:
                coverd_qa_pair += 1
                if 2 in [len(p) for p in statement_paths]:
                    pass
                    # print([len(p) for p in statement_paths])
                qa_path_lenths.append(sum([len(p) for p in statement_paths]) / len(statement_paths))
            path_counts.append(len(statement_paths))
        if len(qa_path_lenths) == 0:  # none qa pair
            if labels[index]:
                no_qa_pair[1] += 1
            else:
                no_qa_pair[0] += 1
            continue
        if labels[index]:
            correct_avg_path_lens_perQA.append(sum(qa_path_lenths) / len(qa_pairs))
            correct_path_counts.append(sum(path_counts)/ len(qa_pairs))
            correct_none_qa_pair_coverage.append(float(coverd_qa_pair) / len(qa_pairs))
        else:
            wrong_avg_path_lens_perQA.append(sum(qa_path_lenths) / len(qa_pairs))
            wrong_path_counts.append(sum(path_counts)/ len(qa_pairs))
            wrong_none_qa_pair_coverage.append(float(coverd_qa_pair) / len(qa_pairs))

    print(no_qa_pair)

    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go

    plotly.tools.set_credentials_file(username='xxxx', api_key='xxxxxx')


    ######
    trace1 = go.Histogram(
        x=correct_avg_path_lens_perQA,
        opacity=0.75,
        histnorm='percent',
        name='correct answers (0.15)',
        nbinsx=20
    )
    trace2 = go.Histogram(
        x=wrong_avg_path_lens_perQA,
        opacity=0.75,
        histnorm='percent',
        name='wrong answers',
        nbinsx=20
    )

    data = [trace1, trace2]
    layout = go.Layout(
        # barmode='stack',
        title="%s: Avg Path Length Per QA-Pair"%flag,
        xaxis=dict(
            title='Avg Path Length'
        ),
        yaxis=dict(
            title='Percentage'
        ),
        bargap=0.2,
        bargroupgap=0.1
    )
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='%s_avg_path_per_qa_pair_%s' % (flag, str(threshold)))

    ######

    trace1 = go.Histogram(
        x=correct_none_qa_pair_coverage,
        opacity=0.75,
        histnorm='percent',
        name='correct answers',
        nbinsx=20
    )
    trace2 = go.Histogram(
        x=wrong_none_qa_pair_coverage,
        opacity=0.75,
        histnorm='percent',
        name='wrong answers',
        nbinsx=20
    )

    data = [trace1, trace2]
    layout = go.Layout(
        # barmode='stack',
        title="%s: None QA-Pair Coverage" % (flag),
        xaxis=dict(
            title='Avg Path Length'
        ),
        yaxis=dict(
            title='Percentage'
        ),
        bargap=0.2,
        bargroupgap=0.1
    )
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='%s_none_qa_pair_coverage %s' % (flag, str(threshold)))

    ######

    trace1 = go.Histogram(
        x=correct_path_counts,
        opacity=0.75,
        histnorm='percent',
        name='correct answers',
        nbinsx=30
    )
    trace2 = go.Histogram(
        x=wrong_path_counts,
        opacity=0.75,
        histnorm='percent',
        name='wrong answers',
        nbinsx=30
    )

    data = [trace1, trace2]
    layout = go.Layout(
        # barmode='stack',
        title="%s: Path Counts" % flag,
        xaxis=dict(
            title='Path Counts'
        ),
        yaxis=dict(
            title='Percentage'
        ),
        bargap=0.2,
        bargroupgap=0.1
    )
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='%s_path_counts_%s' % (flag, str(threshold)))



pathfinding_analysis(PF_PATH, statement_json_file, flag)