'''
Contains source for running DeepDiv experiments.

Based on pytorch_lightning
'''

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from torchvision.transforms import ToTensor

from deepdiv_models import DeepDivResNet
from torchvision.models import resnet18
from torchmetrics.classification import MulticlassAccuracy

# Constants

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
LR = 1e-3
NUM_GPUS = 1
EPOCHS = 40

class DeepDivLearner(pl.LightningModule):
    '''
    Pytorch_Lightning interface for training
    '''

    def __init__(self, model):
        super(DeepDivLearner, self).__init__()
        self.learner = model.to(device)
        self.acc = MulticlassAccuracy(num_classes=self.learner.n_classes).cuda()

    def training_step(self, batch,_):
        return self.learner(batch)
    
    def test_step(self, batch,_):
        x,y = batch
        y_hat = self.learner.infer_by_committee(x)
        acc = self.acc(y_hat, y.cuda())
        self.log('accuracy', acc, on_epoch=True)
        return acc

    def configure_optimizers(self):
        '''
        To keep the number of heads flexible, we need to manually
        add each model's params to the optimizer.
        '''
        parameters = []
        for net in self.learner.models:
            parameters.extend(list(net.parameters()))
        adam = torch.optim.SGD(parameters, lr=LR, momentum=0.9)
        return {'optimizer':adam}


## Training

train_dataset = CIFAR10(root='/research/cwloka/projects/dpitt/data', download=True, train=True, transform=ToTensor())
val_dataset = CIFAR10(root='/research/cwloka/projects/dpitt/data', download=True, train=False, transform=ToTensor())
train_loader = DataLoader(train_dataset, num_workers = 4 * NUM_GPUS, persistent_workers=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, num_workers = 4 * NUM_GPUS, persistent_workers=True, batch_size=BATCH_SIZE)

model = DeepDivResNet(backbone=resnet18(), n_heads=5)
learner = DeepDivLearner(model)

trainer = pl.Trainer(
                accelerator='gpu',
                devices=NUM_GPUS,
                max_epochs=EPOCHS,
                accumulate_grad_batches=1,
                sync_batchnorm=True,
                log_every_n_steps=10,
            )
trainer.fit(model=learner, train_dataloaders=train_loader)
trainer.test(model=learner, dataloaders=val_loader)