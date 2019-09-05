from OpenKE.config import Config
from OpenKE import models
import numpy as np
import tensorflow as tf
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('opt_method', help='SGD/Adagrad/...')
parser.add_argument('pretrain', help='0/1', type=int)
args = parser.parse_args()


def init_predict(hs, ts, rs):

    '''
    # (1) Set import files and OpenKE will automatically load models via tf.Saver().
    con = Config()


    # con.set_in_path("OpenKE/benchmarks/FB15K/")
    con.set_in_path("openke_data/")
    # con.set_test_link_prediction(True)
    con.set_test_triple_classification(True)
    con.set_work_threads(8)
    con.set_dimension(100)


    # con.set_import_files("OpenKE/res/model.vec.tf")
    con.set_import_files("openke_data/embs/glove_initialized/glove.transe.SGD.pt")
    con.init()
    con.set_model(models.TransE)
    con.test()

    con.predict_triple(hs, ts, rs)

    # con.show_link_prediction(2,1)
    # con.show_triple_classification(2,1,3)
    '''

    # (2) Read model parameters from json files and manually load parameters.
    con = Config()
    con.set_in_path("./openke_data/")
    con.set_test_triple_classification(True)
    con.set_work_threads(8)
    con.set_dimension(100)
    con.init()
    con.set_model(models.TransE)
    f = open("./openke_data/embs/glove_initialized/glove.transe.SGD.vec.json", "r")
    content = json.loads(f.read())
    f.close()
    con.set_parameters(content)
    con.test()

    # (3) Manually load models via tf.Saver().
    # con = config.Config()
    # con.set_in_path("./benchmarks/FB15K/")
    # con.set_test_flag(True)
    # con.set_work_threads(4)
    # con.set_dimension(50)
    # con.init()
    # con.set_model(models.TransE)
    # con.import_variables("./res/model.vec.tf")
    # con.test()


def run():

    opt_method = args.opt_method
    int_pretrain = args.pretrain
    if int_pretrain == 1:
        pretrain = True
    elif int_pretrain == 0:
        pretrain = False
    else:
        raise ValueError('arg "pretrain" must be 0 or 1')


    # Download and preprocess ConcepNet

    config = Config()
    config.set_in_path("./openke_data/")
    config.set_log_on(1)  # set to 1 to print the loss

    config.set_work_threads(30)
    config.set_train_times(1000)  # number of iterations
    config.set_nbatches(512)  # batch size
    config.set_alpha(0.001)  # learning rate

    config.set_bern(0)
    config.set_dimension(100)
    config.set_margin(1.0)
    config.set_ent_neg_rate(1)
    config.set_rel_neg_rate(0)
    config.set_opt_method(opt_method)

    '''revision starts'''
    config.set_pretrain(pretrain)

    # Save the graph embedding every {number} iterations

    # OUTPUT_PATH = "./openke_data/embs/glove_initialized/"
    if pretrain:
        OUTPUT_PATH = "./openke_data/embs/glove_initialized/glove."
    else:
        OUTPUT_PATH = "./openke_data/embs/xavier_initialized/"

    '''revision ends'''

    # Model parameters will be exported via torch.save() automatically.
    config.set_export_files(OUTPUT_PATH + "transe."+opt_method+".tf", steps=500)
    # Model parameters will be exported to json files automatically.
    # (Might cause IOError if the file is too large)
    config.set_out_files(OUTPUT_PATH + "transe."+opt_method+".vec.json")

    print("Opt-method: %s" % opt_method)
    print("Pretrain: %d" % pretrain)
    config.init()
    config.set_model(models.TransE)

    print("Begin training TransE")

    config.run()


if __name__ == "__main__":
    # run()
    init_predict(2, 3, 5)
