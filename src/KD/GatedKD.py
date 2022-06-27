import torch
import torch.nn as nn
import torch.nn.functional as F
from KD import BaseClass

class GatedKD(BaseClass):
    """
    Original implementation of Knowledge distillation from the paper "Distilling the
    Knowledge in a Neural Network" https://arxiv.org/pdf/1503.02531.pdf

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module):  Calculates loss during distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer_teacher,
        optimizer_student,
        scheduler_student=None,
        loss_fn=nn.MSELoss(),
        temp=20.0,
        distil_weight=0.5,
        gating_weight=0.01,
        gate_layers=[],
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        super(GatedKD, self).__init__(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            scheduler_student,
            loss_fn,
            temp,
            distil_weight,
            device,
            log,
            logdir,
            gate_layers
        )
        self.gating_weight = gating_weight
        self.gate_layers = gate_layers

    def l1g_penalty(self):
        loss = 0
        for g in self.gate_layers :
            values = torch.cat([x.view(-1) for x in g[0].parameters()])
            loss += torch.norm(values, 1)
        loss = loss / len(self.gate_layers)
        return loss        

    def stdg_penalty(self):
        loss = 0
        for g in self.gate_layers :
            values = torch.cat([x.view(-1) for x in g[0].parameters()])
            loss += (1/(torch.std(values)+1e-4)) / 1e2
        loss = loss / len(self.gate_layers)
        return loss

    def kd_penalty(self, rho=0.05):
        loss = 0
        for g in self.gate_layers :
            p_hat = torch.cat([g[0].parameters()])#torch.cat([x.view(-1) for x in g[0].parameters()])
            print(p_hat)
            funcs = nn.Sigmoid()
            p_hat = torch.mean(funcs(p_hat),1)
            p_tensor = torch.Tensor([rho] * len(p_hat)).to(device)
            loss += torch.sum(p_tensor * torch.log(p_tensor) - p_tensor * torch.log(p_hat) + (1 - p_tensor) * torch.log(1 - p_tensor) - (1 - p_tensor) * torch.log(1 - p_hat))
        return loss

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """

        soft_teacher_out = F.softmax(y_pred_teacher / self.temp, dim=1)
        soft_student_out = F.softmax(y_pred_student / self.temp, dim=1)

        # kd_penalty = self.gating_weight * self.kd_penalty(rho=0.05)
        loss = (1 - self.distil_weight) * F.cross_entropy(y_pred_student, y_true)
        loss += (self.distil_weight * self.temp * self.temp) * self.loss_fn(
            soft_teacher_out, soft_student_out
        )
        
        if (self.gating_weight > 0):
            l1_penalty = self.gating_weight * self.l1g_penalty()
            #std_penalty = self.gating_weight * self.stdg_penalty()
            loss += l1_penalty #+ std_penalty

        return loss
