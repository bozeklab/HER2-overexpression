import torch
from torch.utils.data import Sampler
import math


class DistributedWeightedSampler(Sampler):
    """
    Weighted Sampler adapted to be used as Distributed Sampler
    Adapted from https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/8
    """
    def __init__(self, weights, replacement=True, shuffle=False, num_replicas=None, rank=None):
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        """
        self.weights = weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(self.weights.size()[0] * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle


    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(self.weights.size()[0], generator=g).tolist()
        else:
            indices = list(range(self.weights.size()[0]))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples


        # do the weighted sampling
        subsample_balanced_indices = torch.multinomial(self.weights, self.total_size, self.replacement)
        # subsample the indices
        subsample_balanced_indices = subsample_balanced_indices[indices]

        return iter(subsample_balanced_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class PadForUnroll:
    """ """

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

#draft
"""
def weighted_distributed_loader(dataset)
train_dataset = build_image_dataset(config['dataset']['train'], train_transform)
weights = utils.calculate_weights(torch.tensor(train_dataset.df['HER2_FISH_Score'].values))
train_sampler = DistributedWeightedSampler(weights, num_replicas=config['gpus'], rank=proc_index)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)
"""