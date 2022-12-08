'''
Contains source for running experiments.

Based on pytorch_lightning
'''

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from torchvision.transforms import ToTensor

from models import DeepDivResNet
from torchvision.models import resnet18

# Constants

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
LR = 1e-3
NUM_GPUS = 1
EPOCHS = 10

class DeepDivLearner(pl.LightningModule):
    '''
    Pytorch_Lightning interface for training
    '''

    def __init__(self, model):
        super(DeepDivLearner, self).__init__()
        self.learner = model.to(device)

    def training_step(self, batch):
        return self.learner(batch)
    
    def configure_optimizers(self):
        parameters = []
        for net in self.learner.models:
            #print(list(net.parameters()))
            parameters.extend(list(net.parameters()))
        print(f"{parameters=}")
        adam = torch.optim.SGD(parameters, lr=LR, momentum=0.9)
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