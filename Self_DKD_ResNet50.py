import torch
import os
import numpy as np
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import timm
import detectors

warnings.filterwarnings('ignore')

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(1)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR100('cifar-100', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                             ])),
  batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR100('cifar-100', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                             ])),
  batch_size=1000, shuffle=True)

# ------------------------------------------------------------------------------

# Decoupled Knowledge Distillation
class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])
    
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher):
        super(DKD, self).__init__(student, teacher)
        self.ce_loss_weight = 1.0
        self.alpha = 1.0
        self.beta = 8.0
        self.temperature = 4.0
        self.warmup = 20

    def forward_train(self, data, target, epoch):
        logits_student = self.student(data)
        with torch.no_grad():
            logits_teacher = self.teacher(data)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(epoch / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict

# ------------------------------------------------------------------------------

# Training
t_network = timm.create_model("resnet50_cifar100", pretrained=True)
s_network = timm.create_model("resnet50_cifar100", pretrained=False)

s_network.cuda()
t_network.cuda()

def test():
  s_network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.cuda(), target.cuda()
      output = s_network(data)
      test_loss += F.cross_entropy(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  print(f"Accuracy: {correct / len(test_loader.dataset):.2f}")

num_epochs = 200

distiller = DKD(s_network, t_network)
optimizer = optim.Adam(distiller.parameters(), lr=0.005)
total_steps = len(train_loader) * num_epochs
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

def train():
  s_network.train()
  for epoch in range(1, num_epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()

      data, target = data.cuda(), target.cuda()
      _, losses_dict = distiller.forward_train(data, target, epoch)
      total_loss = sum([l.mean() for l in losses_dict.values()])
      total_loss.backward()
      optimizer.step()
      scheduler.step()

      if batch_idx % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), total_loss.item()))

      if batch_idx % 500 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Learning Rate: {current_lr:.6f}')
        test()
        s_network.train()

train()
test()
