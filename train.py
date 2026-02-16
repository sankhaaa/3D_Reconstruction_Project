# train.py
import gc
import os
import yaml
import time
import random
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm



from data.dataset import MRIReconstructionDataset  # (your dataset file)
from models.unet3d import UNet3D  # your 3D reconstruction model

from utils.metrics import psnr, mae, ssim_3d

val_dataset = MRIReconstructionDataset("data/val")


val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=4
)


def validate(model, val_loader, device):
    model.eval()          # evaluation mode
    total_psnr = 0.0
    val_mae  = 0.0
    val_ssim = 0.0

    with torch.no_grad(): # no gradients
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            total_psnr += psnr(outputs, targets)
            val_mae  += mae(outputs, targets)
            val_ssim += ssim_3d(outputs, targets)

    return total_psnr / len(val_loader), val_mae / len(val_loader), val_ssim / len(val_loader)


def set_seed(seed: int = 42):
    """Ensure reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model weights"""
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, save_path)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.MSELoss()

    dataset = MRIReconstructionDataset("data/train")
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    num_epochs = 20
    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0.0
        train_psnr=0.0
        train_mae  = 0.0
        train_ssim = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            print(f" Batch {i+1}/{len(train_loader)}")
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
              train_psnr += psnr(outputs, targets)
              train_mae  += mae(outputs, targets)
              train_ssim += ssim_3d(outputs, targets)

        avg_loss = total_loss / len(train_loader)
        avg_train_psnr = train_psnr / len(train_loader)
        avg_mae   = train_mae / len(train_loader)
        avg_ssim  = train_ssim / len(train_loader)

        print(f" Epoch {epoch+1} finished | Avg Loss: {total_loss/len(train_loader):.6f}")

        # Free memory after every epoch
        torch.cuda.empty_cache()
        gc.collect()

        val_psnr, val_mae, val_ssim = validate(model, val_loader, device)
        print(
            f"Epoch {epoch+1} | "
            f"Loss: {avg_loss:.6f} | "
            f"Train PSNR: {avg_train_psnr:.2f} | "
            f"Val PSNR: {val_psnr:.2f}"
            f"MAE: {avg_mae:.6f} | "
            f"SSIM: {avg_ssim:.4f}"
        )
    print(" Training complete!")



ds = MRIReconstructionDataset("data/train")
print(f"Total files: {len(ds)}")

for i in range(3):
    vol, _ = ds[i]
    print(f"File {i}: shape={vol.shape}, min={vol.min()}, max={vol.max()}")

if __name__ == "__main__":
    print("🚀 Starting training...")
    train()
model = UNet3D().to("cpu")
torch.save(model.state_dict(), "checkpoints/model_final.pth")
