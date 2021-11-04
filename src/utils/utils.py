import numpy as np
import torch
from torchvision import transforms
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
        x_off, y_off = 0,0
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
    return img_patches, valid_mask_indices 

def calculate_weights(targets):
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    weight = targets.size()[0] / class_sample_count.double()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight

def load_model_without_ddp(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    checkpoint['model'] = new_state_dict
    model.load_state_dict(checkpoint['model'])
    return model
