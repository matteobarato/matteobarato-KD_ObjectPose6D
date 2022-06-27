import os
from tabnanny import verbose
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from src,models.resnet import resnet18
from src,NAS import NASGatedConv

class ResNet18(pl.LightningModule):
    def __init__(self, fully_conv=True, pretrained=True, output_stride=8, remove_avg_pool_layer=False, 
    use_gates_with_penalty=0, lr=1e-3, optim='adam', scheduler_t_max=100):
        super().__init__()
        self.lambda_gates_penalty = use_gates_with_penalty
        if self.lambda_gates_penalty:
            self.NAS = NASGatedConv(verbose=False)

        self.model = resnet18(fully_conv=fully_conv,
                               pretrained=pretrained,
                               output_stride=output_stride,
                               remove_avg_pool_layer=remove_avg_pool_layer)
        self.criterion =  nn.CrossEntropyLoss()
        self.scheduler_t_max = 100
        self.lr = lr
        self.lr_min = self.lr / self.scheduler_t_max
        self.optim = optim
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x_hat = self(x)

        loss = self.criterion(x_hat, y)

        # Optional Gate Penalty
        if (self.lambda_gates_penalty):
            gates_penalty = torch.sum(torch.Tensor([torch.norm(g[0].weight,1) for g in self.NAS.gate_layers]))
            loss += self.lamba_gates_penalty * gates_penalty
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return {"loss": loss, "preds": x_hat}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, y)
        preds = torch.argmax(x_hat, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        return {"loss": loss, "preds": x_hat}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, y)
        preds = torch.argmax(x_hat, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
        return {"loss": loss, "preds": x_hat}


    def configure_optimizers(self):
        if (self.optim=='sgd'):
             optimizer = optim.SGD(self.parameters(), lr=self.lr,
                      momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = optim.Adam(self.parameters(), weight_decay=5e-4, lr=self.lr)

        scheduler_1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=(1/self.scheduler_t_max), end_factor=1.0, total_iters=10)
        scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.scheduler_t_max, eta_min=self.lr_min)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler_1, scheduler_2], [10])
        return [optimizer], [scheduler]

    def apply_gates(self):
        assert self.NAS, 'NAS is not initialized, cannot apply gates to model.'
        return self.NAS.apply_gates_to_model(self.model, apply_to_layer_types=[nn.Conv2d])    

    def estimate_required_channels(self, use_mean=0.1):
        assert self.NAS, 'NAS is not initialized, cannot apply estimate required channels.'
        return self.NAS.estimate_required_channels(use_mean=use_mean) 

    def optimize(self, use_mean=False, amount=0.4):
        return self.NAS.optimize(self.model, use_mean=use_mean, amount=amount)
  

