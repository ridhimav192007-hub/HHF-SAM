import numpy as np
import nibabel as nib
import torch
# Create a fake 3D CT scan
image = np.random.rand(128, 128, 32)

# Create an empty segmentation mask
mask = np.zeros((128, 128, 32))

# Add a fake lesion/organ area to the mask
mask[45:80, 45:80, 15:20] = 1

# Save them as medical NIfTI files
nib.save(
    nib.Nifti1Image(image, np.eye(4)),
    "data/images/sample_image.nii.gz"
)

nib.save(
    nib.Nifti1Image(mask, np.eye(4)),
    "data/masks/sample_mask.nii.gz"
)

print("Sample CT image created!")
print("Sample mask created!")
# Load the CT image and mask
ct = nib.load("data/images/sample_image.nii.gz").get_fdata()
seg = nib.load("data/masks/sample_mask.nii.gz").get_fdata()

print("CT shape:", ct.shape)
print("Mask shape:", seg.shape)
# Choose one slice from the CT scan
slice_number = 17

import cv2

# Take one 2D slice
ct_slice = ct[:, :, slice_number]
mask_slice = seg[:, :, slice_number]

# Resize to 224 x 224
ct_resized = cv2.resize(ct_slice, (224, 224))
mask_resized = cv2.resize(
    mask_slice,
    (224, 224),
    interpolation=cv2.INTER_NEAREST
)

print("Resized CT shape:", ct_resized.shape)
print("Resized mask shape:", mask_resized.shape)
# Normalize CT image between 0 and 1
ct_normalized = (ct_resized - ct_resized.min()) / (
    ct_resized.max() - ct_resized.min()
)

print("Minimum value:", ct_normalized.min())
print("Maximum value:", ct_normalized.max())
# Convert to PyTorch tensors
ct_tensor = torch.tensor(ct_normalized, dtype=torch.float32)
mask_tensor = torch.tensor(mask_resized, dtype=torch.float32)

print("CT tensor shape:", ct_tensor.shape)
print("Mask tensor shape:", mask_tensor.shape)
# Add channel dimension
ct_tensor = ct_tensor.unsqueeze(0)
mask_tensor = mask_tensor.unsqueeze(0)

print("Final CT shape:", ct_tensor.shape)
print("Final mask shape:", mask_tensor.shape)