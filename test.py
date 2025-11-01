import torch
from models.unet3d import UNet3D
from data.dataset import load_nifti, save_nifti
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D()
model.load_state_dict(torch.load("checkpoints/model_final.pth", map_location=device))
model.to(device)
model.eval()

input_folder = "data/test"
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

with torch.no_grad():
    for f in os.listdir(input_folder):
        if not f.endswith(".nii") and not f.endswith(".nii.gz"):
            continue
        vol = load_nifti(os.path.join(input_folder, f))
        vol = vol - vol.min()
        vol = vol / max(vol.max(), 1e-8)
        inp = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
        out = model(inp)
        out_np = out.squeeze().cpu().numpy()
        save_nifti(out_np, os.path.join(output_folder, "recon_" + f),reference_path=vol)
        print(f"✅ Reconstructed {f}")

print("🏁 All reconstructions complete!")
