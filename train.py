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

# -----------------------------
# 1️⃣  Imports: Dataset + Model
# -----------------------------

from data.dataset import MRIReconstructionDataset  # (your dataset file)
from models.unet3d import UNet3D  # your 3D reconstruction model



# -----------------------------
# 2️⃣  Utility Functions
# -----------------------------
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


# -----------------------------
# 3️⃣  Training Function
# -----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    dataset = MRIReconstructionDataset("data/train")
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    num_epochs = 1
    for epoch in range(num_epochs):
        print(f"\n🚀 Epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            print(f"🧩 Batch {i+1}/{len(train_loader)}")
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1} finished | Avg Loss: {total_loss/len(train_loader):.6f}")

        # Free memory after every epoch
        torch.cuda.empty_cache()
        gc.collect()

    print("🏁 Training complete!")


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

