import numpy as np
import torch
import torch.nn as nn
import random
from . import measure
import traceback

def count_parameters(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e3

def cal_regular_factor(model, mu, sigma):

    model_params = torch.as_tensor(count_parameters(model))/1e3
    print('PARAMS',model_params)
    regular_factor =  torch.exp(-(torch.pow((model_params-mu),2)/sigma))
    print('REGULAR FACTOR',regular_factor)
   
    return regular_factor

@measure('swap_svd',bn=True)
def get_score(model, inputs, targets, loss_fn=None, split_data=1):
    #print('SWAPSUIRONALDO')
    model.zero_grad()
    with open("log_12345_SWAP_SVD.txt", "w") as log:
        try:
            swap = SWAP(model=model, inputs=inputs, device='cuda', seed=123, regular=False, mu=None, sigma=None)
            swap.reinit()
            swap_score=swap.forward()
        except Exception as e:
            traceback.print_exc(file=log)
            swap_score = 12345

    #model = model.apply(network_weight_gaussian_init)
    
    
    #print('SWAP_SVD:', swap_score)
    return swap_score

class SampleWiseActivationPatterns(object):
    def __init__(self, device):
        self.swap = -1 
        self.activations = None
        self.device = device

    @torch.no_grad()
    def collect_activations(self, activations):
        n_sample = activations.size()[0]
        n_neuron = activations.size()[1]

        if self.activations is None:
            self.activations = torch.zeros(n_sample, n_neuron).to(self.device)  

        self.activations = torch.sign(activations)

    @torch.no_grad()
    def calSWAP(self, regular_factor):
        
        self.activations = self.activations.T # transpose the activation matrix: (samples, neurons) to (neurons, samples)
        self.swap = torch.unique(self.activations, dim=0).size(0)
        
        del self.activations
        self.activations = None
        torch.cuda.empty_cache()
        #print(self.swap)

        return self.swap * regular_factor


class SWAP:
    def __init__(self, model=None, inputs = None, device='cuda', seed=0, regular=True, mu=1.5, sigma=1.5):
        self.model = model
        self.interFeature = []
        self.seed = seed
        self.regular_factor = 1
        self.inputs = inputs
        self.device = device
        self.result_list=[]

        if regular and mu is not None and sigma is not None:
            self.regular_factor = cal_regular_factor(self.model, mu, sigma).item()
            #print(self.regular_factor)

        self.reinit(self.model, self.seed)

    def reinit(self, model=None, seed=None):
        if model is not None:
            self.model = model
            self.register_hook(self.model)
            self.swap = SampleWiseActivationPatterns(self.device)

        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        self.swap = SampleWiseActivationPatterns(self.device)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for n, m in model.named_modules():
            #if isinstance(m, nn.ReLU):
            m.register_forward_hook(hook=self.hook_in_forward)
            m.register_forward_hook(hook=self.forward_hook)

    def forward_hook(self,module, data_input, data_output):
        measure='dextr'
        #result_list = []
        fea = data_output[0].clone().detach()
        n=torch.tensor(fea.shape[0])
        fea = fea.reshape(n, -1)
        if measure == 'dextr':


            s=torch.linalg.svdvals(fea)
            svd=torch.min(s)/torch.max(s)
            svd[torch.isnan(svd)] = 0
            svd[torch.isinf(svd)] = 0
            self.result_list.append(svd)
            """corr = torch.corrcoef(fea)
            corr[torch.isnan(corr)] = 0
            corr[torch.isinf(corr)] = 0
            values = torch.linalg.eig(corr)[0]
            result = torch.min(torch.real(values))
            self.result_list.append(result)"""



        elif measure == 'dextr_opt':
            idxs = random.sample(range(n), 8)
            fea = fea[idxs, :]
            #corr = torch.corrcoef(fea)
            s=torch.linalg.svdvals(fea)
            svd=torch.min(s)/torch.max(s)
            corr[torch.isnan(svd)] = 0
            corr[torch.isinf(svd)] = 0
            #values = torch.linalg.eig(corr)[0]
            result = svd * n / 8

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach()) 

    def forward(self):
        self.interFeature = []
        with torch.no_grad():
            self.model.forward(self.inputs.to(self.device))
            result_list_svd=torch.tensor(self.result_list)
            result_list_svd = result_list_svd[torch.logical_not(torch.isnan(result_list_svd))]
            #results_svd = torch.square(torch.sum(result_list_svd))
            results_svd = torch.square(torch.sum(result_list_svd))
            #print(results_svd.item())
            if len(self.interFeature) == 0: return
            activtions = torch.cat([f.view(self.inputs.size(0), -1) for f in self.interFeature], 1)         
            self.swap.collect_activations(activtions)
            swap=self.swap.calSWAP(self.regular_factor)/1e4
            print('SWAP: ',swap)
            svd=results_svd.item()
            print('SVD: ',svd)
            dextr= 2 * swap* svd / (swap+svd)
            print('Dextr: ',dextr)
            #dextr= 2 * self.swap.calSWAP(self.regular_factor)* results_svd.item() / (self.swap.calSWAP(self.regular_factor)+results_svd.item())
            return dextr







