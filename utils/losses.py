import torch
import torch.nn as nn




def mse_loss(pred, target):
  return nn.MSELoss()(pred, target)




def l1_loss(pred, target):
  return nn.L1Loss()(pred, target)
