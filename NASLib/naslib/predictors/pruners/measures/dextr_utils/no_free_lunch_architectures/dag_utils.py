import numpy as np
import models

def find_all_paths(Aff, all_paths, all_paths_idx, curr_path=[], curr_path_idx=[], curr_pos=0, end_pos=5):
    if curr_pos == end_pos:
        all_paths.append(list(curr_path))
        all_paths_idx.append(list(curr_path_idx))
        return

    next_nodes = np.where(Aff[curr_pos, (curr_pos+1):] >= 0)[0] + curr_pos + 1
    # print(curr_pos, next_nodes)
    for node in next_nodes:
        curr_path.append(Aff[curr_pos, node])
        curr_path_idx.append([curr_pos, node])
        find_all_paths(Aff, all_paths, all_paths_idx, curr_path, curr_path_idx, node, end_pos)
        curr_path.pop(-1)
        curr_path_idx.pop(-1)
    return all_paths, all_paths_idx


def effective_depth_width(Aff):
    paths, paths_idx = find_all_paths(Aff, [], [], end_pos=len(Aff)-1)
    depth = 0
    width = 0
    depth = 0
    param_edges = [] # num. real effective parameterized edges!
    for path, path_idx in zip(paths, paths_idx):
        depth += np.sum(path)
        width += int(np.sum(path) > 0)
        for node, node_idx in zip(path, path_idx):
            if node == 1:
                param_edges.append("-".join([str(i) for i in node_idx]))
    if depth == 0: return 0, 0, 0
    else:
        depth = depth / len(paths)
        # return depth, width/depth, len(set(param_edges))
        return depth, len(paths), len(set(param_edges))


def dag2affinity(dag):
    # e.g. "2_02_002" => [[2], [0, 2], [0, 0, 2]]
    num_nodes = len(dag) + 1
    Aff = np.ones((num_nodes, num_nodes)) * -1 # from x to
    np.fill_diagonal(Aff, 0)
    for _idx in range(len(dag)):
        to_node = _idx + 1
        edges = dag[_idx]
        for from_node, edge in enumerate(edges):
            Aff[from_node, to_node] = edge - 1 # here -1 is 0, 0 is 1, 1 is 2
    return Aff


import copy
def find_all_dags(all_dags, curr_dag=[], max_num_nodes=4, candidate_ops=[0, 1, 2]):
    # node#0 is omitted
    if len(curr_dag) == max_num_nodes-1 and len(curr_dag[-1]) == max_num_nodes-1:
        all_dags.append(copy.deepcopy(list(curr_dag)))
        return

    if len(curr_dag) == 0 or len(curr_dag[-1]) == len(curr_dag):
        curr_dag.append([])
    for op in candidate_ops:
        curr_dag[-1].append(op)
        find_all_dags(all_dags, curr_dag, max_num_nodes, candidate_ops)
        curr_dag[-1].pop(-1)
    if len(curr_dag[-1]) == 0:
        curr_dag.pop(-1)
    return all_dags


def build_model(args, classes=10, dummy_shape=(3, 32, 32)):
    if args.arch.startswith("resnet"):
        model = models.__dict__[args.arch](pretrained=args.pretrained, num_classes=classes, imagenet=args.imagenet_arch)
    elif args.arch == "mlp":
        #  by default use mlp_dags
        model = models.mlp_dags(args.dag.split('-'), np.prod(dummy_shape), args.width, out_dim=classes, bn=args.bn, bias=args.bias,
                                supernet=getattr(args, "supernet", False),
                                separated_readout=getattr(args, "separated_readout", False),
                                lora_rank=getattr(args, "lora", -1)) #
    elif args.arch == "convnet":
        model = models.convnet(args.dag, dummy_shape[0], args.width, out_dim=classes, bn=args.bn, fc=args.fc)
    return model
