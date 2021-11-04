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
def calculate_sample_weights(targets):
    num_samples = targets.size
    classes_count = np.bincount(targets)
    num_classes = classes_count.size
    class_weights = num_samples / (num_classes * classes_count)
    sample_weights = [class_weights[t] for t in targets]
    return sample_weights


#Class to register a hook on the target layer (used to get the output channels of the layer)
class Hook():
    #register_full_backward_hook has issues for complex modules but should be fixed
    def __init__(self, module, hook_fn = None):
        self.hook_forward = module.register_forward_hook(self.forward_fn)
        self.hook_backward = module.register_full_backward_hook(self.backward_fn)
        self.hook_fn = hook_fn
    
    def forward_fn(self, module, input, output):
        self.input = input
        self.output = output
        if self.hook_fn != None:
            self.input, self.output = self.hook_fn(self.input, self.output)
        #self.input.retain_grad()
        #self.output.retain_grad()
    
    def backward_fn(self, module, grad_input, grad_output):
        self.grad_input = grad_input
        self.grad_output = grad_output
        if self.hook_fn != None:
            self.grad_input, self.grad_output = \
                self.hook_fn(self.grad_input, self.grad_output)


class GradCAM():
    def __init__(self, model, activations_layer):
        self.model = model
        self.activations_layer_hook = Hook(activations_layer)
    
    def make_CAM(self, input_tensor, c):
        self.model.zero_grad()
        out = self.model(input_tensor)
        loss = out[0,c]
        loss.backward()
        
        num_features = self.activations_layer_hook.output.shape[1]
        a_k_c = self.activations_layer_hook.grad_output[0][0].sum(dim = (1,2)) / num_features
        A_k = self.activations_layer_hook.output[0]
        L_c = A_k * a_k_c.unsqueeze(0).unsqueeze(0).permute(2,1,0)
        L_c = torch.nn.functional.relu(L_c.sum(dim = 0))
        self.model.zero_grad()
        return L_c
    
    def close(self):
        self.hook_forward.remove()
        self.hook_backward.remove()


class ResNet_HER2_FISH(torch.nn.Module):
    def __init__(self):
        super(ResNet_HER2_FISH, self).__init__()
        self.resnet = models.resnet34(pretrained = False, num_classes = 4)
        self.hook = Hook(self.resnet.avgpool)
        self.second_fc = torch.nn.Linear(512, 2)


    def forward(self, x):
        out_her2 = self.resnet(x)
        out_avgpool = self.hook.output
        out_avgpool = torch.flatten(out_avgpool, 1)
        out_fish = self.second_fc(out_avgpool)
        return (out_her2, out_fish)

def get_preds_and_true_labels(df, prob_cols, true_col):
    predicted_probs = df[prob_cols].values
    predicted_probs = softmax(predicted_probs, axis = 1)
    predicted_labels = np.argmax(predicted_probs, axis = 1)
    true_labels = df[true_col].values
    return predicted_labels, true_labels

def plot_confusion_matrix(confusion_matrix, labels, fig_size = (10, 7), title = None, ax = None, annot = None):
    confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=labels, index = labels)
    confusion_matrix_df.index.name = 'Actual'
    confusion_matrix_df.columns.name = 'Predicted'
    if annot is None:
        annot = True
    else:
        
        annot_df = pd.DataFrame(annot, columns=labels, index = labels)
        annot_df.index.name = 'Actual'
        annot_df.columns.name = 'Predicted'
        annot = annot_df       
    sn.set(font_scale=1.4)#for label size
    if ax == None:
        plt.figure(figsize = fig_size)
        hm =sn.heatmap(confusion_matrix_df, cmap="Blues",annot_kws={"size": 16}, fmt='g', annot = annot)# font size
    else:
        hm = sn.heatmap(confusion_matrix_df, cmap="Blues", annot_kws={"size": 16}, fmt='g', ax = ax, annot = annot)
    hm.set_title(title)

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

def merge_csvs(csvs_list, filename):
    dataframes_list = list(map(lambda x : pd.read_csv(x), csvs_list ) )
    pd.concat(dataframes_list).to_csv(filename, index = False)


def load_yaml(yaml_path):
    """
    Loads a json config from a file.
    """
    assert os.path.exists(yaml_path), "Json file %s not found" % yaml_path
    yaml_file = open(yaml_path)
    yaml_config = yaml_file.read()
    yaml_file.close()
    try:
        config = yaml.safe_load(yaml_config)
    except BaseException as err:
        raise Exception("Failed to validate config with error: %s" % str(err))

    return config

def get_name_and_kwargs(dict):
    dict = dict.copy()
    name = dict.pop('name', None)
    kwargs = dict
    return name, kwargs



def build_summary_writers(dir, proc_index):
    if proc_index == 0 :
        train_writer = SummaryWriter(log_dir = os.path.join(dir, 'logs/Scalars/Train'))
        val_writer = SummaryWriter(log_dir = os.path.join(dir, 'logs/Scalars/Val'))
    else:
        train_writer = None
        val_writer = None

    return train_writer, val_writer

def log_phase_results(summary_writer, dict, epoch):
    for item in dict.items():
        summary_writer.add_scalar(*item, epoch)


def load_checkpoint(checkpoint_dir, model, optimizer = None):
    checkpoint = torch.load(checkpoint_dir, map_location = torch.device('cuda')) #checkpointeo solo funciona para gpu!!
    model.load_state_dict(checkpoint['model'])    
    epoch0 = checkpoint['epoch'] + 1
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return epoch0
