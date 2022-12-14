'''
Contains source for running DeepDiv experiments.

Based on pytorch_lightning
'''

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.models import resnet18
from torchmetrics import Accuracy

from bnn_models import BayesianResNet
from ..utils import pairwise_inner_product

# Constants

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
LR = 1e-3
NUM_GPUS = 1
EPOCHS = 10

class BayesianLearner(pl.LightningModule):
    '''
    Pytorch_Lightning interface for training
    '''

    def __init__(self, model, mc_samples:int = 10):
        super(BayesianLearner, self).__init__()
        self.learner = model.to(device)
        self.mc_samples = mc_samples
        self.acc = Accuracy(task='multiclass', num_classes=self.learner.n_classes)
        # TODO: replace this with catching all representations for pairwise comparisons
        self.representations = None 


    def training_step(self, batch):
        return self.learner(batch)
    
    def test_step(self, batch):
        x,y = batch
        representations = []
        y_hats = torch.empty((self.mc_samples, y.size()[0]))
        for i in range(self.mc_samples):
            y_hat, phi = self.learner.infer(x)
            y_hats[i,:] = y_hat
            representations.append(phi)
        #y_hat = torch.mean(y_hats, dim=0)
        y_hat_var, y_hat = torch.var_mean(y_hats,unbiased=False, dim=0)
        acc = self.acc(y_hat, y)
        similarity = pairwise_inner_product(representations=representations)
        for 



    def configure_optimizers(self):
        adam = torch.optim.SGD(self.learner.parameters(), lr=LR, momentum=0.9)
        return {'optimizer':adam}


## Training

dataset = CIFAR10(root='/research/cwloka/projects/dpitt/data', download=True, train=True, transform=ToTensor())
loader = DataLoader(dataset, num_workers = 4 * NUM_GPUS, persistent_workers=True, batch_size=BATCH_SIZE)

model = DeepDivResNet(backbone=resnet18(), n_heads=3, device=device)
learner = DeepDivLearner(model)

trainer = pl.Trainer(
                accelerator='gpu',
                devices=NUM_GPUS,
                max_epochs=EPOCHS,
                accumulate_grad_batches=1,
                sync_batchnorm=True,
                log_every_n_steps=10,
            )
trainer.fit(model=learner, train_dataloaders=loader)