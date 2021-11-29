import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import src.utils as utils

class GatedAttention(nn.Module):
    """
    computes a vector z which is the sum of the input feature vectores weighted by their attention.
    modified from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
    "Attention-based Deep Multiple Instance Learning" by Ilsen et al. 
    """
    def __init__(self, input_size, hidden_size):
        super(GatedAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.attention_V = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        z = torch.mm(A, x)  # KxL
        return z, A

    
class ResnetABMIL(nn.Module):
    def __init__(self, hidden_size = 256, patch_size = 224, freeze_resnet = False, pretrained = False, progress = True, **kwargs):
        super(ResnetABMIL, self).__init__()


        resnet = models.resnet34(pretrained, progress, **kwargs)        
        if freeze_resnet:
            for param in resnet.parameters():
                param.requires_grad = False
            
        resnet_modules = list(resnet.children())
        self.patch_size = patch_size
        self.feature_extractor = nn.Sequential(*resnet_modules[:-1])
        #resnet34 last layer vectors have 512 features
        self.attention_mechanism = GatedAttention(512, hidden_size)
        #if pretrained = True the classifier is also going to be pretrained, and it only works for num_classes = 1000 (imagenet)
        self.classifier = nn.Linear(512, kwargs.get('num_classes'))
    
    def get_valid_patches(self, x):
        rand_offset = True if self.training else False
        x, valid_ind = utils.get_valid_patches(x, self.patch_size, self.patch_size, rand_offset = rand_offset)
        return x, valid_ind
    
    def forward(self, x):
        #model only allows batch size = 1
        if x.shape[0] != 1:
            raise ValueError('Model only admits batch size = 1!')

        x = x.squeeze(0)
        x, valid_ind = self.get_valid_patches(x)
        x = self.feature_extractor(x)
        x, A = self.attention_mechanism(x)
        pred = self.classifier(x)
        return (pred, x, A, valid_ind)
