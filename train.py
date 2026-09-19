import torch
import torch.optim as optim

from models.med_adapter import sam, mhpm, hfm, phfd, features
from loss import SegmentationLoss


# Collect only trainable parameters
trainable_params = []

for model in [sam, mhpm, hfm, phfd]:
    trainable_params += [
        p for p in model.parameters()
        if p.requires_grad
    ]


# Loss function
criterion = SegmentationLoss()


# AdamW optimizer
optimizer = optim.AdamW(
    trainable_params,
    lr=0.001,
    weight_decay=0.1
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.1
)

print("Training components initialized successfully!")
print("Number of trainable parameter tensors:", len(trainable_params))
# Training function
# Training loop
def train_model(dataloader, epochs=50):

    for epoch in range(epochs):

        total_loss = 0.0

        for images, targets in dataloader:

            optimizer.zero_grad()

            # Clear previous features
            features.clear()

            # Forward through SAM
            _ = sam.image_encoder(images)

            # Process features using MHPM
            processed_features = []

            for layer in [3, 6, 9, 12]:

                feature = features[layer].permute(0, 3, 1, 2)
                feature = mhpm(feature)
                processed_features.append(feature)

            # Fuse multi-scale features
            fused_features = sum(processed_features) / len(
                processed_features
            )

            # Hierarchical Feature Module
            fused_features = hfm(fused_features)

            # PHFD decoder
            predictions = phfd(fused_features)

            # Match target size with prediction
            targets = torch.nn.functional.interpolate(
                targets.float(),
                size=predictions.shape[-2:],
                mode="nearest"
            )

            # Calculate loss
            loss = criterion(predictions, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss / len(dataloader):.4f}"
        )

        # Update learning rate
        scheduler.step()