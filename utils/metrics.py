import torch
import math
import torch.nn.functional as F




def psnr(pred, target, max_val=1.0):
  mse = torch.mean((pred - target) ** 2)
  if mse == 0:
    return float('inf')
  return 20 * torch.log10(max_val / torch.sqrt(mse))




def mae(pred, target):
  return torch.mean(torch.abs(pred - target)).item()



def ssim_3d(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool3d(x, 3, 1, 1)
    mu_y = F.avg_pool3d(y, 3, 1, 1)

    sigma_x = F.avg_pool3d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool3d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool3d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return (ssim_n / ssim_d).mean()

def dice_score(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

 