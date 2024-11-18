import argparse
import pickle
import random

from foresight.dataset import *
from foresight.models import nasbench2
from foresight.weight_initializers import init_net
from models import get_cell_based_tiny_net
from nas_201_api import NASBench201API as API
from scipy.stats import entropy
import torch
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from tqdm import tqdm
import numpy as np
from no_free_lunch_architectures.length import * #for curvature
import scipy
from nats_bench import create

def get_score(net, x, device, measure='dextr'):
    result_list = []
    max_svd_list=[]
    meco_list_min=[]
    meco_list_max=[]
    entropy_list_min = []
    entropy_list_max = []
    kl_divs=[]
    svds=[]
    n_list=[]
    kappa_list=[]
    svd_list=[]
    bs=128

    def forward_hook(module, data_input, data_output):
        #svds=[]
        #result_list = []
        fea = data_output[0].clone().detach()
        shape=fea.shape

        n=torch.tensor(fea.shape[0])
        #n1=torch.tensor(prev_fea.shape[0])
        fea = fea.reshape(n, -1)
        #print('FEA',fea.shape)

        if measure == 'dextr':

            s=torch.linalg.svdvals(fea)
            svd=torch.min(s)/torch.max(s)
            #print('Cond number ', svd)
            svd[torch.isnan(svd)] = 0
            svd[torch.isinf(svd)] = 0
            result_list.append(svd)

    hooks=[]
    for name, modules in net.named_modules():

        hooks.append(modules.register_forward_hook(forward_hook))
        #f+=1
    x = x.to(device)
    net(x)
    for hook in hooks:
        hook.remove()

    result_list = torch.tensor(result_list)
    result_list = result_list[torch.logical_not(torch.isnan(result_list))]
    svd= torch.sum(result_list)
    svd=torch.log(1+svd)
    try:
        curvature=get_extrinsic_curvature(net,size_curve=(1, 3, 64, 64))
        curvature=torch.log(1+torch.tensor(curvature))
    except Exception as e:
        print('Curvature Error')
        return svd.item(), result_list, max_svd_list
    if np.isnan(curvature):
        print('CURVATURE IS NAN')
        return svd.item(), result_list, max_svd_list
    
    results = svd *curvature / (svd + curvature)

    if (torch.isnan(results) or torch.isinf(results)):
        return svd.item(), result_list, max_svd_list

    print('SVD: ',svd)
    print('Curvature: ',curvature)
    print('Dextr: ',results.item())
    return results.item(), result_list, max_svd_list
    #return (results_min.item(), entropies_min.item(), results_max.item(), entropies_max.item())

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120


def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201')
    # parser.add_argument('--api_loc', default='../data/NAS-Bench-201-v1_0-e61699.pth',
    #                     type=str, help='path to API')
    parser.add_argument('--outdir', default='output/',
                        type=str, help='output directory')
    parser.add_argument('--search_space', type=str, default='tss', choices=['tss', 'sss'])
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero, one]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero, one]')
    parser.add_argument('--measure', type=str, default='dextr', choices=['dextr'])
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--data_size', type=int, default=32, help='data_size')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='appoint', help='random, grasp, appoint supported')
    parser.add_argument('--dataload_info', type=int, default=1,
                        help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=15625, help='end index')
    parser.add_argument('--noacc', default=False, action='store_true',
                        help='avoid loading NASBench2 api an instead load a pickle file with tuple (index, arch_str)')
    parser.add_argument('--only_curvatures', default=False, action='store_true',
                        help='Get only curvatures')
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args

def calculate_spearman(path,num_archs):
    val_accs=[]
    dextrs=[]
    test_accs=[]
    for i in range(0,num_archs):
        with open(path + str(i),'rb') as f:
            x=pickle.load(f)
        val_accs.append(x['valacc'])
        dextrs.append(x['dextr'])
        test_accs.append(x['testacc'])
    return scipy.stats.spearmanr(test_accs,dextrs)

if __name__ == '__main__':
    args = parse_arguments()
    print(args.device)

    

    api = create(None, args.search_space, fast_mode=True, verbose=False)
    #api = API('/home/asthana/Documents/ZCProxy/nas201_data/NAS-Bench-201-v1_0-e61699.pth')

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers, resize=args.data_size)
    x, _ = next(iter(train_loader))
    y = torch.randint(0, get_num_classes(args), (args.batch_size,))
    print(x.shape, y.shape)
    cached_res = []
    cached_feats=[]
    cached_kappa=[]
    cached_maxsvd=[]
    pre = 'cf' if 'cifar' in args.dataset else 'im'
    pfn = f'nb2_{args.search_space}_{pre}{get_num_classes(args)}_seed{args.seed}_dl{args.dataload}_dlinfo{args.dataload_info}_initw{args.init_w_type}_initb{args.init_b_type}_{args.batch_size}.p'
    op = os.path.join(args.outdir, pfn)

    end = len(api) if args.end == 0 else args.end

    # loop over nasbench2 archs
    for i, arch_str in tqdm(enumerate(api)):

        if i < args.start:
            continue
        if i >= end:
            break

        res = {'i': i, 'arch': arch_str}
        if args.search_space == 'tss':
            net = nasbench2.get_model_from_arch_str(arch_str, get_num_classes(args))
            arch_str2 = nasbench2.get_arch_str_from_model(net)
            #print(arch_str2)
            if arch_str != arch_str2:
                print(arch_str)
                print(arch_str2)
                raise ValueError
        elif args.search_space == 'sss':
            config = api.get_net_config(i, args.dataset)
            net = get_cell_based_tiny_net(config)
        net.to(args.device)

        init_net(net, args.init_w_type, args.init_b_type)

        if (args.only_curvatures):
            curv=get_extrinsic_curvature(net,size_curve=(128, 3, 32, 32))
            with open(f'curvatures_nats_{args.search_space}_final.txt','a') as f:
                f.write(str(curv))
                f.write('\n')
        else:

            measures, result_list,max_svd_list= get_score(net, x, args.device, measure=args.measure)
            #print(measures)

            res[f'{args.measure}'] = measures

            if not args.noacc:
                if args.search_space == 'tss':
                    info = api.get_more_info(i, 'cifar10-valid' if args.dataset == 'cifar10' else args.dataset, iepoch=None,
                                            hp='200', is_random=False)
                else:
                    info = api.get_more_info(i, 'cifar10-valid' if args.dataset == 'cifar10' else args.dataset, iepoch=None,
                                            hp='90', is_random=False)

                trainacc = info['train-accuracy']
                valacc = info['valid-accuracy']
                testacc = info['test-accuracy']

                res['trainacc'] = trainacc
                res['valacc'] = valacc
                res['testacc'] = testacc

            #print(res)
            cached_res.append(res)

            # write to file
            if i % args.write_freq == 0 or i == len(api) - 1 or i == 10:
                #print(f'writing {len(cached_res)} results to {op}')
                pf = open(op+str(i), 'ab')
                for cr in cached_res:
                    pickle.dump(cr, pf)
                pf.close()
                cached_res = []
    


    # Calculate Spearman rank correlation coeeficient
    print('Spearman rank correlation coefficient: ', calculate_spearman(path=op,num_archs=args.end-args.start)[0])