import numpy as np
import torch


def get_ntk_nngp_eig(Xs, network, train_mode=False):
    device = torch.cuda.current_device()
    ntks = []
    if train_mode:
        network.train()
    else:
        network.eval()
    ######
    grads = []
    inputs = Xs.cuda(device=device, non_blocking=True)
    network.zero_grad()

    features, logits = network(inputs, return_all=True)
    nngp = torch.einsum('nc,mc->nm', [features[-1], features[-1]])
    eigenvalues_nngp, _ = torch.symeig(nngp)  # ascending
    if isinstance(logits, tuple):
        logits = logits[1]
    for _idx in range(len(inputs)):
        logits[_idx:_idx+1].backward(torch.ones_like(logits[_idx:_idx+1]), retain_graph=True) # gradients from logits
        grad = []
        for name, W in network.named_parameters():
            if 'weight' in name or 'bias' in name:
                if W.grad is not None:
                    grad.append(W.grad.view(-1).detach())
        grads.append(torch.cat(grad, -1))
        network.zero_grad()
        torch.cuda.empty_cache()
    ######
    grads = torch.stack(grads, 0)
    ntk = torch.einsum('nc,mc->nm', [grads, grads])
    eigenvalues_ntk, _ = torch.symeig(ntk)  # ascending
    return eigenvalues_nngp, eigenvalues_ntk
