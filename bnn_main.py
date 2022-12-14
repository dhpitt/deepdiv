'''
Contains source for running DeepDiv experiments.

Based on pytorch_lightning
'''

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.models import resnet18
from torchmetrics.classification import MulticlassAccuracy

from bnn_models import BayesianResNet
from numeric_utils import get_avg_similarity_per_example

# Constants

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
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
        self.acc = MulticlassAccuracy(num_classes=self.learner.n_classes).cuda()
        # TODO: replace this with catching all representations for pairwise comparisons
        self.representations = None 


    def training_step(self, batch, _):
        '''
        boilerplate training step
        '''
        return self.learner(batch)
    
    def test_step(self, batch, _):
        '''
        Compute uncertainty using traditional variance (as in Kendall + Gal)
            and using average cosine similarity (my new metric)
        '''
        x,y = batch
        representations = torch.empty((self.mc_samples, x.size()[0], self.learner.representation_dim)).cuda()
        y_hats = torch.empty((self.mc_samples, y.size()[0])).cuda()
        for i in range(self.mc_samples):
            logits, phi = self.learner.infer(x)
            p_hat = torch.softmax(logits, dim=1)
            y_hat = torch.argmax(p_hat, dim=1)
            y_hats[i,:] = y_hat
            representations[i, :, :] = phi
        y_hat_var, y_hat = torch.var_mean(y_hats,unbiased=True, dim=0)
        acc = self.acc(y_hat, y)
        similarity = get_avg_similarity_per_example(representations=representations)
        corr = F.cosine_similarity(y_hat_var, similarity, dim=0)
        print(f'{corr=}')

        return acc


    def configure_optimizers(self):
        opt = torch.optim.SGD(self.learner.parameters(), lr=LR, momentum=0.9)
        return {'optimizer':opt}


## Training

train_set = CIFAR10(root='/research/cwloka/projects/dpitt/data', download=True, train=True, transform=ToTensor())
test_set = CIFAR10(root='/research/cwloka/projects/dpitt/data', download=True, train=False, transform=ToTensor())
train_loader = DataLoader(train_set, num_workers = 4 * NUM_GPUS, persistent_workers=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, num_workers = 4 * NUM_GPUS, persistent_workers=True, batch_size=BATCH_SIZE)

model = BayesianResNet(model=resnet18(), n_classes=10)
learner = BayesianLearner(model)

trainer = pl.Trainer(
                accelerator='gpu',
                devices=NUM_GPUS,
                max_epochs=EPOCHS,
                accumulate_grad_batches=1,
                sync_batchnorm=True,
                log_every_n_steps=10,
            )
#trainer.fit(model=learner, train_dataloaders=train_loader)
trainer.test(model=learner, dataloaders=test_loader)