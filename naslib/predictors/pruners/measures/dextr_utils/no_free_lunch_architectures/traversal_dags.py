'''
    collect meta data for DAGs
'''
#!/usr/bin/python
#!python
from distutils.command.build import build
import os
import pdb
import time
import pickle
import random
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
from zmq import CURVE

# from models import model_dict
import models
from dag_utils import effective_depth_width, dag2affinity, find_all_dags

from pdb import set_trace as bp

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training Subnetworks')

##################################### General setting ############################################
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='./experiment', type=str)
parser.add_argument('--exp_name', help='additional names for experiment', default='', type=str)


args = parser.parse_args()


PID = os.getpid()

def main():
    global args, best_acc

    job_name = "TRAVERSAL-DAG{exp_name}".format(
        exp_name="" if args.exp_name == "" else "-"+args.exp_name)
    timestamp = "{:}".format(time.strftime("%m%d%H%M%S", time.gmtime()))
    args.save_dir = os.path.join(args.save_dir, job_name, timestamp)

    os.makedirs(args.save_dir, exist_ok=True)

    OP_SPACE = [0, 1, 2]
    ALL_DAGS = find_all_dags([], max_num_nodes=4, candidate_ops=OP_SPACE)
    ALL_DAGS_valid = []

    print("<< ============== (PID = %d) %s ============== >>"%(PID, args.save_dir))
    for dag in tqdm(ALL_DAGS):
        depth, width, depth_total = effective_depth_width(dag2affinity(dag))
        ALL_DAGS_valid.append([dag, depth, width, depth_total]) # #parameterized edges
        if depth == 0:
            continue

    np.save(os.path.join(args.save_dir, "dags_valid.npy"), ALL_DAGS_valid)


if __name__ == '__main__':
    main()
