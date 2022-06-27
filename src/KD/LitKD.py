import os
from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch import nn, optim
import torch.nn.functional as F

class BaseClass (pl.LightningModule):
    def __init__(self, 
        teacher_model, 
        student_model, 
        loss_fn=nn.KLDivLoss(), 
        temp=20.0,
        distill_weight=0.5, verbose=False) -> None:
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.loss_fn = loss_fn
        self.temp = temp
        self.distill_weight = distill_weight
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        return self.student_model(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x_hat_teacher = self.teacher_model.predict_step(batch, batch_idx)
        student_out = self.student_model.training_step(batch, batch_idx)
        x_hat_student = student_out['preds']
        loss = self.calculate_kd_loss(x_hat_student, x_hat_teacher, student_out['loss'],  y)

        # Logging to TensorBoard by default
        self.log("train_kd_loss", loss)
        if self.student_model.NAS:
            self.log("train_channels_mean",  [torch.mean(w[0].transformed_weight(), dim=0).item() for w in  self.student_model.NAS.gate_layers])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat_teacher = self.teacher_model.predict_step(batch, batch_idx)
        student_out = self.student_model.validation_step(batch, batch_idx)
        x_hat_student = student_out['preds']
        loss = self.calculate_kd_loss(x_hat_student, x_hat_teacher, student_out['loss'],  y)
        preds = torch.argmax(x_hat_student, dim=1)

        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat_teacher = self.teacher_model.predict_step(batch, batch_idx)
        student_out = self.student_model.test_step(batch, batch_idx)
        x_hat_student = student_out['preds']
        loss = self.calculate_kd_loss(x_hat_student, x_hat_teacher, student_out['loss'],  y)
        preds = torch.argmax(x_hat_student, dim=1)

        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)


    def configure_optimizers(self):
        self.student_model.configure_optimizers()

    def calc_improvement(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())

        print("-" * 80)
        print("Total parameters for the teacher network are: {}".format(teacher_params))
        print("Total parameters for the student network are: {}".format(student_params))
        print("Parameters improvement: {}".format(student_params/teacher_params))


    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, loss_student, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        raise NotImplementedError



class GatedKD(BaseClass):
    def __init__(self, 
        teacher_model, 
        student_model, 
        loss_fn=nn.KLDivLoss(), 
        temp=20.0,
        distill_weight=0.5, verbose=False):
        super(GatedKD, self).__init__(
            teacher_model,
            student_model,
            loss_fn,
            temp,
            distill_weight,
            verbose
            )
        
    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, loss_student, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """
        soft_teacher_out = F.softmax(y_pred_teacher / self.temp, dim=1)
        soft_student_out = F.softmax(y_pred_student / self.temp, dim=1)

        # kd_penalty = self.gating_weight * self.kd_penalty(rho=0.05)
        loss = (1 - self.distill_weight) * loss_student
        loss += (self.distill_weight * self.temp * self.temp) * self.loss_fn(
            soft_teacher_out, soft_student_out
        )
        
        return loss
