import torch
import torch.nn as nn
import loralib as lora
from pdb import set_trace as bp


def exclude_prune(model):
    # protect from pruning
    for name, m in model.named_modules():
        if not hasattr(m, "_exclude_prune"):
            setattr(m, "_exclude_prune", True)


def add_shared_count(model):
    # protect from pruning
    for name, m in model.named_modules():
        if not hasattr(m, "_shared_count"):
            setattr(m, "_shared_count", 1)
        else:
            setattr(m, "_shared_count", getattr(m, "_shared_count") + 1)


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, lora_rank=-1, bn=False, bias=True):
        super(Block, self).__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._lora_rank = lora_rank
        self._bn = bn
        self._bias = bias
        layer = []
        if lora_rank > 0:
            layer.append(lora.Linear(in_dim, out_dim, r=lora_rank, bias=bias))
        else:
            layer.append(nn.Linear(in_dim, out_dim, bias=bias))
        if bn: layer.append(nn.BatchNorm1d(out_dim))
        layer.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*layer)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layer(x)


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x * 0


# OP: 0 "zero", 1 "skip", 2 "linear"
class MLP(nn.Module):
    def __init__(self, dags, in_dim, width, out_dim=10,
                 share_edge=False, supernet=False, separated_readout=False, lora_rank=-1,
                 bn=False, bias=True):
        super(MLP, self).__init__()
        self._dags = []
        self._in_dim = in_dim
        self._width = width
        self._out_dim = out_dim
        self._share_edge = share_edge # share weight for samge edge type across dags
        self._supernet = supernet # merge node (sum up features)
        self._separated_readout = separated_readout
        self._lora_rank = lora_rank
        self._bn = bn
        self._bias = bias
        self.register_buffer('alphas', torch.ones(len(dags)))
        self._stem = Block(in_dim, width, bn=bn, bias=bias)
        exclude_prune(self._stem)
        self._readout = nn.Linear(width, out_dim, bias=bias)
        exclude_prune(self._readout)
        # _dags: _dag, _to, _from
        self._to_from_dag_layers, self._dags = self._build_dag_layers(dags)
        self._init()

    def _build_dag_layers(self, dags):
        for _idx, _dag in enumerate(dags): # _dag, _to, _from
            # list of to_node (list of in_node). 0: broken; 1: skip-connect; 2: linear or conv
            # e.g. "2_02_002" => [[2], [0, 2], [0, 0, 2]]
            if isinstance(_dag, str):
                _dag = [[int(edge) for edge in node] for node in _dag.split('_')]
                dags[_idx] = _dag
            elif isinstance(_dag, list):
                assert isinstance(_dag[0], list) and len(_dag[0]) == 1 # 2nd node has one in-degree
                for i in range(1, len(_dag)):
                    assert len(_dag[i]) == len(_dag[i-1]) + 1 # next node has one more in-degree than prev node
        _to_from_dag_layers = nn.ModuleList() # _to, _from, _dag
        for _to in range(len(dags[0])):
            _to_from_dag_layers.append(nn.ModuleList())
            for _from in range(len(dags[0][_to])):
                _to_from_dag_layers[-1].append(nn.ModuleList())
                for _idx in range(len(dags)):
                    if self._share_edge and _idx > 0 and dags[_idx][_to][_from] == dags[_idx-1][_to][_from]:
                        # share the same operator as prev dag
                        _to_from_dag_layers[-1][-1].append(_to_from_dag_layers[-1][-1][-1])
                    else:
                        _to_from_dag_layers[-1][-1].append(self._build_layer(dags[_idx][_to][_from]))
                    add_shared_count(_to_from_dag_layers[-1][-1][-1]) # for pruning
        return _to_from_dag_layers, dags

    def _build_layer(self, op):
        if op == 2:
            return Block(self._width, self._width, self._lora_rank, self._bn, bias=self._bias)
        elif op == 1:
            return nn.Identity()
        else:
            return Zero()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)

    def _set_zeros(self):
        # for LoRA, set original weight as zeros
        for _to in range(len(self._dags[0])):
            for dag_idx, _alpha in enumerate(self.alphas):
                for _from in range(len(self._dags[0][_to])):
                    if isinstance(self._to_from_dag_layers[_to][_from][dag_idx], Block):
                        self._to_from_dag_layers[_to][_from][dag_idx].layer[0].weight.data.fill_(0.)

    def _get_dag_layers(self, dag_idx):
        # for LoRA, set original weight as zeros
        layers = []
        for _to in range(len(self._dags[0])):
            for _from in range(len(self._dags[0][_to])):
                layers.append(self._to_from_dag_layers[_to][_from][dag_idx])
        return layers

    def shift_alphas(self, delta, bottom=0.5, top=1.5):
        for dag_idx in range(len(self.alphas)):
            if self.alphas[dag_idx] > 0:
                if dag_idx == len(self.alphas) - 1:
                    assert self.alphas[dag_idx] == 1
                else:
                    self.alphas[dag_idx] = max(bottom, self.alphas[dag_idx] - delta)
                    self.alphas[dag_idx+1] = min(top, self.alphas[dag_idx+1] + delta)
                return

    def _copy_dag_weights(self, dag_idx1, dag_idx2):
        # copy from dag1 to dag2
        for _to2 in range(len(self._dags[0])):
            for _from2 in range(len(self._dags[0][_to2])):
                if self._dags[dag_idx2][_to2][_from2] != 2: continue
                # iteratively match each weights (op = 2) for dag_idx2
                matched = False
                for _delta_from in range(max(_from2, len(self._dags[0][_to2]) - _from2) + 1):
                    # first prefer the same _from
                    for _from1 in [_from2, _from2 + _delta_from, _from2 - _delta_from]:
                        for _delta_to in range(max(_to2, len(self._dags[0]) - _to2) + 1):
                            for _to1 in [_to2, _to2 + _delta_to, _to2 - _delta_to]:
                                if _to1 < 0 or _to1 >= len(self._dags[0]) or _from1 < 0 or _from1 >= len(self._dags[0][_to1]): continue
                                if self._dags[dag_idx2][_to2][_from2] == self._dags[dag_idx1][_to1][_from1]:
                                    if not (_from1 == _from2 and _to1 == _to2):
                                        self._to_from_dag_layers[_to2][_from2][dag_idx2].load_state_dict(self._to_from_dag_layers[_to1][_from1][dag_idx1].state_dict())
                                    matched = True
                                if matched: break
                            if matched: break
                        if matched: break
                    if matched: break

    def forward_single(self, x):
        nodes = []
        for dag_idx in range(len(self._dags)):
            _nodes = [x] # output from prev node, input to next node
            for _to in range(len(self._dags[0])):
                _node = []
                for _from in range(len(self._dags[0][_to])):
                    _node.append(self._to_from_dag_layers[_to][_from][dag_idx](_nodes[_from]))
                _nodes.append(sum(_node))
            nodes.append(_nodes[-1])
        if self._separated_readout:
            return nodes, [self._readout(out) for out in nodes]
        else:
            return nodes, self._readout(sum(nodes))

    def forward_supernet(self, x):
        nodes_to = [[[1, x]]] # list of node's feature => a list of (alpha, output)-pair from prev node (by each DAG), input to next node
        for _to in range(len(self._dags[0])):
            _node_to_dag = [] # list of [alpha, dag's output at this "to"]
            for dag_idx, _alpha in enumerate(self.alphas):
                _node_to_dag_from = []
                for _from in range(len(self._dags[0][_to])):
                    if _alpha == 0: continue
                    _node_to_dag_from.append(self._to_from_dag_layers[_to][_from][dag_idx](sum(a * out for a, out in nodes_to[_from])))
                _node_to_dag.append([_alpha, sum(_node_to_dag_from)])
            nodes_to.append(_node_to_dag)
        if self._separated_readout:
            return nodes_to, [_alpha * self._readout(out) for _alpha, out in nodes_to[-1]]
        else:
            return nodes_to, self._readout(sum([out for _, out in nodes_to[-1]]))

    def forward(self, x, return_all=False):
        x = torch.flatten(x, 1)
        x = self._stem(x)
        if self._supernet:
            feature, output = self.forward_supernet(x)
        else:
            feature, output = self.forward_single(x)
        if return_all:
            return feature, output
        else:
            return output
