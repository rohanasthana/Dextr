import numpy as np
import torch
from torch.nn.functional import normalize
from torch import autograd
# from utils import bn_resume_tracking, bn_stop_tracking
from pdb import set_trace as bp


def get_curve_input(size_curve):
    n_interp, CHW = size_curve[0], size_curve[1:]
    theta = torch.linspace(0, 2 * np.pi, n_interp).cuda(non_blocking=True)
    theta.requires_grad_(True)
    curve_input = torch.matmul(torch.svd(torch.randn(np.prod(CHW), 2).cuda(non_blocking=True))[0], torch.stack([torch.cos(theta), torch.sin(theta)])).T.reshape((n_interp, *CHW))
    curve_input.requires_grad_(True)
    return theta, curve_input



def get_extrinsic_curvature(network, curve_inputs=None, size_curve=None, batch_size=256, train_mode=True):
    if curve_inputs:
        theta, curve_input = curve_inputs
    else:
        theta, curve_input = get_curve_input(size_curve)
    kappa = 0
    network = network.cuda()
    if train_mode:
        network.train()
    else:
        network.eval()
    network.zero_grad()
    _idx = 0
    #print(curve_input.shape)
    while _idx < len(curve_input):
        output = network(curve_input[_idx:_idx+batch_size])
        #output=output[1]
        output = output.reshape(output.size(0), -1)
        n, c = output.size()
        v_s = [] # 1st derivative
        a_s = [] # 2nd derivative
        c_list = [np.random.randint(0, c) for _ in range(10)] #extract 10 random coordinates
        for index in c_list:
            v = autograd.grad(output[:, index].sum(), theta, create_graph=True, retain_graph=True)[0][_idx:_idx+batch_size] # batch size (of thetas)
            a = autograd.grad(v.sum(), theta, create_graph=True, retain_graph=True)[0][_idx:_idx+batch_size] # batch size (of thetas)
            v_s.append(v.detach().clone())
            a_s.append(a.detach().clone())
        v_s = torch.stack(v_s, 0).permute(1, 0) # batch_size x c
        a_s = torch.stack(a_s, 0).permute(1, 0) # batch_size x c
        vv = torch.einsum('nd,nd->n', v_s, v_s)
        aa = torch.einsum('nd,nd->n', a_s, a_s)
        va = torch.einsum('nd,nd->n', v_s, a_s)
        #print(vv.shape, aa.shape, va.shape)
        kappa += (vv**(-3/2) * (vv * aa - va ** 2).sqrt()).sum().item()
        #print(kappa)
        #torch.cuda.empty_cache()
        _idx += batch_size
    torch.cuda.empty_cache()
    return torch.tensor(kappa)

def get_extrinsic_curvature_opt(network, curve_inputs=None, size_curve=None, batch_size=256, train_mode=True):
    if curve_inputs:
        theta, curve_input = curve_inputs
    else:
        theta, curve_input = get_curve_input(size_curve)
    kappa = 0
    network = network.cuda()
    if train_mode:
        network.train()
    else:
        network.eval()
    network.zero_grad()
    _idx = 0
    #print(curve_input.shape)
    while _idx < len(curve_input):
        output = network(curve_input[_idx:_idx+batch_size])
        #output=output[1]
        output = output.reshape(output.size(0), -1)
        n, c = output.size()
        v_s = [] # 1st derivative
        a_s = [] # 2nd derivative
        c_list = [np.random.randint(0, c) for _ in range(10)]
        for index in c_list:
            v = autograd.grad(output[:, index].sum(), theta, create_graph=True, retain_graph=True)[0][_idx:_idx+batch_size] # batch size (of thetas)
            a = autograd.grad(v.sum(), theta, create_graph=True, retain_graph=True)[0][_idx:_idx+batch_size] # batch size (of thetas)
            v_s.append(v.detach().clone())
            a_s.append(a.detach().clone())
        v_s = torch.stack(v_s, 0).permute(1, 0) # batch_size x c
        a_s = torch.stack(a_s, 0).permute(1, 0) # batch_size x c
        vv = torch.einsum('nd,nd->n', v_s, v_s)
        aa = torch.einsum('nd,nd->n', a_s, a_s)
        va = torch.einsum('nd,nd->n', v_s, a_s)
        #print(vv.shape, aa.shape, va.shape)
        kappa += (vv**(-3/2) * (vv * aa - va ** 2).sqrt()).sum().item()
        #print(kappa)
        #torch.cuda.empty_cache()
        _idx += batch_size
    torch.cuda.empty_cache()
    return torch.tensor(kappa)



def curve_complexity_differentiable(network, curve_inputs=None, size_curve=None, batch_size=32, train_mode=True, need_graph=True, reduction='mean',
                                    differentiable=False):
    # network.apply(bn_stop_tracking)
    if curve_inputs:
        theta, curve_input = curve_inputs
    else:
        theta, curve_input = get_curve_input(size_curve)
    LE = 0
    network = network.cuda()
    if train_mode:
        network.train()
    else:
        network.eval()
    network.zero_grad()
    _idx = 0
    while _idx < len(curve_input):
        output = network(curve_input[_idx:_idx+batch_size])
        output = output.reshape(output.size(0), -1)
        _idx += batch_size
        n, c = output.size()
        jacobs = []
        for coord in range(c):
            _gradients = autograd.grad(outputs=output[:, coord].sum(), inputs=[ theta ], only_inputs=True, retain_graph=need_graph, create_graph=need_graph)
            if differentiable:
                jacobs.append(_gradients[0]) # select gradient of "theta"
            else:
                jacobs.append(_gradients[0].detach()) # select gradient of "theta"
        jacobs = torch.stack(jacobs, 0)
        jacobs = jacobs.permute(1, 0)
        gE = torch.einsum('nd,nd->n', jacobs, jacobs).sqrt()
        LE += gE.sum()
        torch.cuda.empty_cache()
    if reduction == 'mean':
        return LE / len(theta)
    else:
        return LE

