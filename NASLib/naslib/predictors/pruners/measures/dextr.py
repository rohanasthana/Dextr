# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
import copy
import time

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import torch
import random
from torch import nn
from scipy.stats import entropy
from .dextr_utils.no_free_lunch_architectures.length import get_extrinsic_curvature, get_extrinsic_curvature_opt, get_curve_input
from . import measure
import traceback

def count_parameters(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e3

def cal_regular_factor(model, mu, sigma):

    model_params = torch.as_tensor(count_parameters(model))/1e3
    print('PARAMS',model_params)
    regular_factor =  torch.exp(-(torch.pow((model_params-mu),2)/sigma))
    #print('REGULAR FACTOR',regular_factor)
    #print(regular_factor)
   
    return regular_factor

def get_score(net, x, device, split_data):
    #print(x.shape)
    #meco_list = []
    #entropy_list = []
    result_list = []
    #x = torch.randn(size=(1, 3, 64, 64)).to(device)
    net.to(device)
    with open("log_12345.txt", "w") as log:
        try:
            curvature=get_extrinsic_curvature_opt(net,size_curve=(1, x.shape[1], x.shape[2], x.shape[3]))
            #curvature=0
        except Exception as e:
            traceback.print_exc(file=log)
            curvature=np.nan

    #scale_curvature = lambda C: C * torch.pow(10, 1 - torch.floor(torch.log10(torch.tensor(C))))
    #curvature = scale_curvature(curvature)
    #print(curvature)


    
    def forward_hook(module, data_input, data_output):
        fea = data_output[0].clone().detach()
        #prev_fea=data_input[0].clone().detach()
        n = torch.tensor(fea.shape[0])
        #n1=torch.tensor(prev_fea.shape[0])
        #print(fea.shape)
        fea = fea.reshape(n, -1)
        #idxs = random.sample(range(n), 8)
        #fea = fea[idxs, :]
        s=torch.linalg.svdvals(fea)
        svd=torch.min(s) / torch.max(s)

        svd[torch.isnan(svd)] = 0
        svd[torch.isinf(svd)] = 0
        #svd=svd*n/8
        result_list.append(svd)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    #print(split_data)
    '''for sp in range(1):
        st = sp * N // 1
        en = (sp + 1) * N // 1
        y = net(x[st:en])'''
    y=net(x)
        # break
    results = torch.tensor(result_list)
    #entropies=torch.tensor(entropy_list)
    results = results[torch.logical_not(torch.isnan(results))]
    #entropies = entropies[torch.logical_not(torch.isnan(entropies))]
    if np.isnan(curvature):
        print('CURVATURE IS NAN')
        results = torch.tensor(torch.sum(results))
        results=torch.log(1+results)
        print('Dextr: ',results.item())
        return results.item()
    else:
        results = torch.tensor(torch.sum(results))#*cal_regular_factor(net, 30, 30)
    #entropies = torch.sum(entropies)
    #meco_list.clear()
    #entropy_list.clear()
    results=torch.log(1+results)
    curvature=torch.log(1+torch.tensor(curvature))
    result_list.clear()
    print('SVD: ',results.item())
    print('Curvature: ',curvature)
    dextr = results *curvature/(results+curvature)
    print('Dextr: ',dextr)
    return dextr.item()



@measure('dextr', bn=True)
def compute_dextr(net, inputs,targets, loss_fn=None,split_data=1 ):
    #print(split_data)
    #size_curve=(1, 3, inputs.shape[2], inputs.shape[3])
    #curve_inputs=theta, curve_input = get_curve_input(size_curve)
    device = 'cuda:0'
    #inputs=torch.tensor(inputs,device='cpu')
    #device='cpu'
    print('INPUTS: ',inputs.shape)
    #curvature=get_extrinsic_curvature(net,curve_inputs)
    #curvature_opt=get_extrinsic_curvature_opt(net,curve_inputs)
    #print('CURVATURE: ',curvature_opt)
    #return curvature_opt.item()
    # Compute gradients (but don't apply them)
    net.zero_grad()
    with open("log_12345.txt", "w") as log:
        try:
            dextr = get_score(net, inputs, device, split_data=split_data)
        except Exception as e:
            traceback.print_exc(file=log)
            dextr = 12345
    return dextr
    '''model_params = torch.as_tensor(count_parameters(net))/1e3
    print('PARAMS',model_params)
    with open('model_size_0_1000.txt','a') as f:
        f.write(str(model_params.item())+'\n')
    return 0'''
