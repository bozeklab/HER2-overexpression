import torch
from torch.utils.data import Sampler
import math


class DistributedWeightedSampler(Sampler):
    """
    Weighted Sampler adapted to be used as Distributed Sampler
    Adapted from https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/8
    """
    def __init__(self, weights, replacement=True, shuffle=False, num_replicas=None, rank=None):
        self.weights = weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(self.weights.size()[0] * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle

    def __iter__(self):
        # g is always created with the same seed
        g = torch.Generator()
        # deterministically shuffle based on epoch (to make torch.multinomial yield different indices)
        if self.shuffle:
              g.manual_seed(self.epoch)
        # do the weighted sampling
        subsample_balanced_indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g)
        # subsample the indices according to rank
        subsample_balanced_indices = subsample_balanced_indices[self.rank:self.total_size:self.num_replicas]
        return iter(subsample_balanced_indices.tolist())

    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch

class PadForUnroll:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, x):
        C, H, W = x.shape
        pad_w = W % self.patch_size
        pad_h = H % self.patch_size
        return torch.nn.functional.pad(
            input=x,
            pad=(pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), 
            mode='constant', value=1.)
