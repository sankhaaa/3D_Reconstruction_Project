import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


orig_path = "../data/test/119.nii"
recon_path = "../results/recon_119.nii"

import os
print("Current working directory:", os.getcwd())

orig = nib.load(orig_path).get_fdata()
recon = nib.load(recon_path).get_fdata()
#
#slice_idx = orig.shape[2] // 2
#
#plt.figure(figsize=(15,5))
#plt.subplot(1,3,1); plt.imshow(orig[:,:,slice_idx], cmap='gray'); plt.title("Original")
#plt.subplot(1,3,2); plt.imshow(recon[:,:,slice_idx], cmap='gray'); plt.title("Reconstructed")
#plt.subplot(1,3,3); plt.imshow(np.abs(orig[:,:,slice_idx]-recon[:,:,slice_idx]), cmap='hot'); plt.title("Difference")
#plt.show()

from nilearn import plotting

view = plotting.view_image(
    recon_path,
    cmap='gray',
    symmetric_cmap=False,
    opacity=1.0,               # fully opaque slices
    threshold=None,            # show full intensity range
    resampling_interpolation='nearest',  # avoid smoothing
    title="Sharpened 3D Reconstruction"
)
view.open_in_browser()
