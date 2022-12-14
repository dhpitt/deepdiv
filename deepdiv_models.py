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
import torchvision.models as tm

from ..numeric_utils import pairwise_CKA
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

        self.feature_extractor = resnet.cuda()
        representation_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        self.classifier = nn.Linear(representation_dim, self.n_classes).cuda()

    def extract_features(self, x):
        return self.feature_extractor(x)
    
    def classify_features(self, phi):
        return self.classifier(phi)


class DeepDivResNet(nn.Module):
    # Model with diverse backbones

    def __init__(self, backbone, n_heads=3, n_classes=10):
        super(DeepDivResNet, self).__init__()
        self.models = [SingleResNetHead(resnet=deepcopy(backbone), n_classes=n_classes) for _ in range(n_heads)]
        self.classification_loss = torch.nn.CrossEntropyLoss()
        self.n_heads = n_heads
        self.n_classes = n_classes

    def forward(self, batch):
        total_loss = 0
        x,y = batch

        x = x.cuda()
        y = y.cuda()

        representations = []
        for model in self.models:
            phi = model.extract_features(x)
            z_hat = model.classify_features(phi) # output logits
            representations.append(phi)
            total_loss += self.classification_loss(input=z_hat, target=y)

        # get pairwise CKA
        total_loss += pairwise_CKA(representations)

        return total_loss
    
    def infer_by_committee(self, x):
        '''
        Batch split up in testing step of PL interface
        '''
        x = x.cuda()

        preds = torch.empty(size=(self.n_heads, x.size()[0], self.n_classes))
        preds = []
        for i,model in enumerate(self.models):
            phi = model.extract_features(x)
            z_hat = model.classify_features(phi) # output logits

            preds[i, :, :] = nn.functional.softmax(z_hat, dim=1)
        p_hat = torch.squeeze(torch.mean(preds, axis=0))
        return torch.argmax(p_hat, dim=1)


if __name__ == '__main__':
    model = DeepDivResNet(backbone=tm.resnet18(), n_heads=4)
    data = torch.randn((10,3,224,224))
    model.train()
    loss = model((data, torch.ones(10,dtype=torch.long)))
    print(loss)