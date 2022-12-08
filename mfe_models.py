'''
Contains source for all models defined under the deep-diversity scheme

Whereas the models in Lee & Finn's paper, "DivDis", shared a feature extractor backbone and had several 
heads meant to capture a diverse subset of the design space for classification heads, my models will not share backbones.
Instead of adding a regularization term to the training objective that maximizes disagreement between heads,
I add a term that maximizes diversity of representations of the information between backbones. 
'''
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as tnnF
import torchvision.models as tm

from utils import CKA

'''
Model convention: 

all models will have two required methods: extract_features and classify_features

extract_features(x) = forward pass x through self.backbone
classify_features(phi) = pass phi through linear layer
'''

class SingleResNetHead(nn.Module):
    # One head model. Backbone and linear layer

    def __init__(self, resnet=tm.resnet18(), n_classes=10):
        super(SingleResNetHead, self).__init__()

        self.n_classes = n_classes

        self.feature_extractor = resnet
        representation_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        self.classifier = nn.Linear(representation_dim, self.n_classes)

    def extract_features(self, x):
        return self.feature_extractor(x)
    
    def classify_features(self, phi):
        return self.classifier(phi)


class DeepDivResNet(nn.Module):
    # Model with diverse backbones

    def __init__(self, backbone, n_heads=3, n_classes=10):
        super(DeepDivResNet, self).__init__()
        self.models = [SingleResNetHead(resnet=backbone, n_classes=n_classes) for _ in range(n_heads)]
        self.classification_loss = torch.nn.CrossEntropyLoss()

        
    def forward(self,x):
        
            


model1 = tm.resnet18(weights=None)
model2 = deepcopy(model1)