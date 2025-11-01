import torch
import math




def psnr(pred, target, max_val=1.0):
  mse = torch.mean((pred - target) ** 2)
  if mse == 0:
    return float('inf')
  return 20 * torch.log10(max_val / torch.sqrt(mse))




def mae(pred, target):
  return torch.mean(torch.abs(pred - target)).item()
