import os
import numpy as np
import itertools
import torch
import torch.nn
from torchvision import models, transforms
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import softmax
from collections import OrderedDict


rng = np.random.default_rng(seed=0)

def get_patches(tensor, tile_size, stride, return_unfold_shape = False):
    dims = tensor.dim()
    tensor_unfold = tensor.unfold(dims-2, size = tile_size, step = stride).unfold(dims-1, size = tile_size, step = stride)
    tensor_patches = tensor_unfold.reshape(*list(tensor.shape)[:-2], -1, tile_size, tile_size)
    if return_unfold_shape:
        return tensor_patches, tensor_unfold.shape
    else:
        return tensor_patches


def calculate_areas(tensor):
    dims = tensor.dim()
    return tensor.sum(dim = dims-2).sum(dim = dims-2)


def get_valid_patches(img_tensor, tile_size, stride, rand_offset = True):
    mask_transform = transforms.Compose([ transforms.Grayscale(), (lambda x: 1 - (x > 220./255)*1) ])
    if rand_offset:
        x_off, y_off = rng.integers(stride), rng.integers(stride)
    else:
        x_off, y_off = 0,0 #esto lo modifique
    img_tensor = img_tensor[..., y_off:, x_off:]
    mask_tensor = mask_transform(img_tensor)
    img_patches = get_patches(img_tensor, tile_size, stride)
    mask_patches = get_patches(mask_tensor, tile_size, stride)
    mask_patches_areas = calculate_areas(mask_patches)
    area_th = 0.05 * tile_size * tile_size
    valid_mask_indices = mask_patches_areas > area_th
    mask_patches = mask_patches[valid_mask_indices].view(*list(mask_patches.shape)[:-3],-1,tile_size, tile_size)
    valid_img_indices = torch.cat(3*[valid_mask_indices], dim = 0)
    img_patches = img_patches[valid_img_indices].view(*list(img_patches.shape)[:-3],-1,tile_size, tile_size)
    img_patches = img_patches.permute(1,0,2,3)
    return img_patches, valid_mask_indices #esto tambien



def valid_patches_generator(img_patches, max_num_patches, batch_size, randomized = True):
    num_tiles = img_patches.shape[0]
    if randomized:
        random_tiles_indices = torch.randperm(num_tiles)
        img_patches = img_patches[random_tiles_indices]

    while num_tiles < max_num_patches :
        img_patches = img_patches.repeat(2,1,1,1)
        num_tiles = img_patches.shape[0]

    for i in range(0, max_num_patches, batch_size):
        bot, top = i, i + batch_size
        if top < num_tiles:
            patches_batch = img_patches[bot:top]
        else:
            patches_batch = img_patches[bot::]
        
        yield patches_batch


def calculate_weights(targets):
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    #weight = 1. / class_sample_count.double()
    weight = targets.size()[0] / class_sample_count.double()
    samples_weight = torch.tensor([weight[t] for t in targets])
    #print(samples_weight)
    return samples_weight

#lo mismo que arriva pero en numpy
# def calculate_sample_weights(targets):
#     num_samples = targets.size
#     classes_count = np.bincount(targets)
#     num_classes = classes_count.size
#     class_weights = num_samples / (num_classes * classes_count)
#     sample_weights = [class_weights[t] for t in targets]
#     return sample_weights

def load_model_without_ddp(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    #model was Distributed Data Parallel, so need to fix that
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    checkpoint['model'] = new_state_dict
    model.load_state_dict(checkpoint['model'])
    return model

def load_checkpoint(checkpoint_dir, model, optimizer = None):
    checkpoint = torch.load(checkpoint_dir, map_location = torch.device('cuda')) #checkpointeo solo funciona para gpu!!
    model.load_state_dict(checkpoint['model'])    
    epoch0 = checkpoint['epoch'] + 1
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return epoch0
