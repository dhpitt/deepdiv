'''
All source for BNN models
'''
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
import torch
from torch import nn

class BayesianResNet(nn.Module):
    '''
    
    '''
    def __init__(self, model, n_classes):
        super(BayesianResNet, self).__init__()
        self.feature_extractor = model
        self.representation_dim = self.model.fc.in_features
        self.n_classes = n_classes

        # Make classifier
        self.model.fc = nn.Identity()
        self.classifier = nn.Linear(self.representation_dim, self.n_classes)
        self.classification_loss = nn.CrossEntropyLoss()

        const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
        }   

        dnn_to_bnn(self.feature_extractor, const_bnn_prior_parameters)
        dnn_to_bnn(self.classifier, const_bnn_prior_parameters)

    def forward(self, batch):
        x,y = batch
        total_loss = 0

        phi = self.feature_extractor(x)
        y_hat = self.classifier(phi)

        total_loss += self.classification_loss(input=y_hat, target=y)
        total_loss += get_kl_loss(self.feature_extractor) / x.size()[0]
        total_loss += get_kl_loss(self.classifer) / x.size()[0]

        return total_loss

    def infer(self, x):
        '''
        Inference. Returns both predictions and feature representations,
        which I'll use down the line to estimate epistemic uncertainty
        based on dissimilarity between samples

        I split the batch into x and y in the PyTorch-lightning interface for the test step.
        '''
        phi = self.feature_extractor(x)
        y_hat = self.classifier(phi)

        return y_hat, phi



                    