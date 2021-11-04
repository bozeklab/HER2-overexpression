from torch.utils.data import Dataset
#import torch
#import math
from PIL import Image
import numpy as np
#import itertools
#import os
import pandas as pd

class ImageDataset(Dataset):
    """Dataset class for a .csv containing images paths
    
    Arguments:

    TBD:

    """

    def __init__(self, df, fn_col = None, lbl_col = None, transform = None, return_filename = False):
        self.df = df
        self.fn_col = fn_col if fn_col != None else df.columns[0]
        self.lbl_col = lbl_col if lbl_col != None else df.columns[1]
        self.transform = transform
        self.return_filename = return_filename

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        fn = self.df.iloc[idx][self.fn_col]
        image = Image.open(fn)
        image = image.convert('RGB')
        if self.transform != None:
            image = self.transform(image)
        lbl = self.df.iloc[idx][self.lbl_col]
        out_tuple = (image, lbl, fn) if self.return_filename else (image, lbl)
        return out_tuple
    
    def df(self):
        return self.df






