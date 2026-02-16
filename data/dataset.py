"""Dataset that loads NIfTI (.nii or .nii.gz) volumes and optionally
returns patches/whole volumes. This implementation returns whole volumes
(simpler). Resize/patch extraction can be added later.
"""


import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import os

import nibabel as nib
import numpy as np
import torch
import os

def save_nifti(volume, path, reference_path=None):
    """
    Saves a numpy array or torch tensor as a NIfTI file.
    Optionally copies affine info from a reference NIfTI file.
    """
    if torch.is_tensor(volume):
        volume = volume.detach().cpu().numpy()

    affine = np.eye(4)
    #if reference_path is not None:
     #   affine = nib.load(reference_path).affine

    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(volume, affine), path)
    print(f"💾 Saved reconstructed volume to: {path}")



def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data

class MRIReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, root, target_shape=(32, 256, 256)):
        self.files = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith('.nii') or f.endswith('.nii.gz')
        ])
        self.target_shape = target_shape   # (D,H,W)
        print(f"📦 Found {len(self.files)} NIfTI volumes in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vol = load_nifti(self.files[idx])

        # normalize 0-1
        vol -= vol.min()
        if vol.max() > 0:
            vol /= vol.max()

        vol = torch.from_numpy(vol).unsqueeze(0).float()   # (1,D,H,W)
        _, D, H, W = vol.shape
        T, Ht, Wt = self.target_shape

        # --- fix depth (D) ---
        if D < T:
            pad_b, pad_a = (T - D)//2, T - D - (T - D)//2
            vol = F.pad(vol, (0,0,0,0,pad_b,pad_a))
        elif D > T:
            start = (D - T)//2
            vol = vol[:, start:start+T, :, :]

        # --- fix height (H) ---
        if H < Ht:
            pad_b, pad_a = (Ht - H)//2, Ht - H - (Ht - H)//2
            vol = F.pad(vol, (0,0,pad_b,pad_a,0,0))
        elif H > Ht:
            start = (H - Ht)//2
            vol = vol[:, :, start:start+Ht, :]

        # --- fix width (W) ---
        if W < Wt:
            pad_b, pad_a = (Wt - W)//2, Wt - W - (Wt - W)//2
            vol = F.pad(vol, (pad_b,pad_a,0,0,0,0))
        elif W > Wt:
            start = (W - Wt)//2
            vol = vol[:, :, :, start:start+Wt]

        assert vol.shape[1:] == self.target_shape, f"shape mismatch {vol.shape}"
        return vol, vol


if __name__ == '__main__':
    ds = MRIReconstructionDataset('data/train')
    print('num samples', len(ds))
    x, y = ds[0]
    print(x.shape, y.shape)
