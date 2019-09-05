# Running Baselines

1. Clone [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) into this directory.
2. Use one of the following scripts to run the specified baseline:

```
run_csqa_bert.sh    (run_csqa_bert.py)
run_csqa_gpt.sh     (pytorch-pretrained-BERT/examples/run_openai_gpt.py)
run_swag_bert.sh    (python pytorch-pretrained-BERT/examples/run_swag.py)
run_swag_gpt.sh     (run_swag_gpt.py)
```
All scripts are set to run with early stopping with a patience of 10 (quits after 10 epochs if no progress was made on the test set).

Recorded performance can be found in this [spreadsheet](https://docs.google.com/spreadsheets/d/1pMVgcTyomzc649LPf1HYwCB0fb_80RH81YjO68LDV9s/edit#gid=0).