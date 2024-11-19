
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from pdb import set_trace as bp

__all__  = ['pruning_model', 'pruning_model_random', 'prune_model_custom', 'remove_prune',
            'extract_mask', 'reverse_mask', 'check_sparsity', 'check_sparsity_dict',
            'pruning_model_random_layer_specified'
            ]


# Pruning operation
def pruning_model(model, px):
    print('Apply Unstructured L1 Pruning Globally (all conv layers)')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def pruning_model_random(model, px, conv1=False):
    print('Apply Unstructured Random Pruning Globally (all conv layers)')
    parameters_to_prune = []
    for name, m in model.named_modules():
        if hasattr(m, '_exclude_prune') and getattr(m, '_exclude_prune') is True:
            # protected parameters
            continue
        if hasattr(m, 'weight'):
            parameters_to_prune.append((m, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )


def pruning_model_random_layer_specified(models, px, conv1=False):
    print('Apply Unstructured Random Pruning Globally (all conv layers), pruning ratio specified per layer')
    parameters_to_prune = []
    if not isinstance(models, list):
        models = [models]
    for model in models:
        for name, m in model.named_modules():
            if hasattr(m, '_exclude_prune') and getattr(m, '_exclude_prune') is True:
                # protected parameters
                continue
            factor = 1.
            if hasattr(m, '_shared_count'):
                # adjust pruning amount due to weight sharing
                factor = getattr(m, '_shared_count')
            amount_to_prune = 1 - (1 - px) * factor #  e.g. if ensemble 2 dags, ideally px should >= 0.5
            assert amount_to_prune >= 0 and amount_to_prune <= 1
            if hasattr(m, 'weight'):
                prune.global_unstructured(
                    tuple([(m, 'weight')]),
                    pruning_method=prune.RandomUnstructured,
                    amount=amount_to_prune,
                )


def prune_model_custom(model, mask_dict):
    print('Pruning with custom mask (all conv layers)')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask_name = name+'.weight_mask'
            if mask_name in mask_dict.keys():
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[mask_name].long())
            else:
                print('Can not find [{}] in mask_dict'.format(mask_name))


def remove_prune(model):
    print('Remove hooks for multiplying masks (all conv layers)')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m,'weight')


# Mask operation function
def extract_mask(model_dict):

    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict

def reverse_mask(mask_dict):

    new_dict = {}
    for key in mask_dict.keys():

        new_dict[key] = 1 - mask_dict[key]

    return new_dict

# Mask statistic function
def check_sparsity(model):

    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        # if isinstance(m, nn.Conv2d):
        if hasattr(m, 'weight'):
            factor = 1.
            if hasattr(m, '_shared_count'):
                # adjust pruning amount due to weight sharing
                factor = getattr(m, '_shared_count')
            assert factor >= 1 and factor == round(factor)
            num_params_this_module = float(m.weight.nelement())
            num_zeros_this_module = float(torch.sum(m.weight == 0))
            # sum_list = sum_list + num_params_this_module * factor
            # zero_sum = zero_sum + (num_params_this_module - (num_params_this_module - num_zeros_this_module) / factor) * factor
            sum_list = sum_list + num_params_this_module
            zero_sum = zero_sum + num_zeros_this_module
            print("{}: #params {}, #zeros {} | total #params {}, #zeros {}".format(name, num_params_this_module, num_zeros_this_module, sum_list, zero_sum))

    if zero_sum:
        remain_weight_rate = 100*(1-zero_sum/sum_list)
        print('* remain weight ratio = ', 100*(1-zero_sum/sum_list),'%')
    else:
        print('no weight for calculating sparsity')
        remain_weight_rate = 100

    if hasattr(m, "_is_shared"):
        setattr(m, "_is_shared", True)

    return remain_weight_rate

def check_sparsity_dict(state_dict):

    sum_list = 0
    zero_sum = 0

    for key in state_dict.keys():
        if 'mask' in key:
            sum_list += float(state_dict[key].nelement())
            zero_sum += float(torch.sum(state_dict[key] == 0))

    if zero_sum:
        remain_weight_rate = 100*(1-zero_sum/sum_list)
        print('* remain weight ratio = ', 100*(1-zero_sum/sum_list),'%')
    else:
        print('no weight for calculating sparsity')
        remain_weight_rate = None

    return remain_weight_rate


