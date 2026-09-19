# HHF-SAM Research Paper Work

## Paper Title
Rectal Cancer Segmentation via HHF-SAM: A Hierarchical Hypercolumn-Guided Fusion Segment Anything Model

## What I understood from the paper

This paper is about segmentation of rectal cancer from CT images.

The main problem is that rectal tumors can be difficult to segment because CT images may have low contrast, noise and unclear boundaries.

The model proposed in the paper is called HHF-SAM.

It mainly has three parts:

1. Med-Adapter SAM Encoder (MSE)
2. Multi-scale Hypercolumn Processing Module (MHPM)
3. Progressive Hierarchical Fusion Decoder (PHFD)

## Work I have done

First, I read and understood the basic architecture of the paper.

Then I started implementing the model using Python and PyTorch.

### SAM-B

I downloaded and loaded the pretrained SAM-B model.

The original SAM parameters were frozen.

### LoRA and Adapter

I added LoRA and Adapter modules to SAM.

LoRA was added to Query and Value.

Adapter modules were also added to the SAM encoder blocks.

### Feature Extraction

I extracted features from SAM encoder layers:

- Layer 3
- Layer 6
- Layer 9
- Layer 12

### MHPM

I implemented the MHPM module for processing features at different scales.

### HFM

I implemented HFM to work with local, regional and global feature information.

### PHFD

I implemented a PHFD-style decoder to generate the final segmentation mask.

### Dataset Preprocessing

I created a sample synthetic CT image and segmentation mask in NIfTI format to test the coding pipeline.

I performed:

- CT slice extraction
- Resizing to 224 × 224
- Normalization
- Conversion to PyTorch tensors

### Training Code

I also implemented the training code.

I used:

- AdamW optimizer
- Learning rate = 0.001
- Weight decay = 0.1
- 50 epochs
- Learning rate scheduler

### Evaluation

I implemented:

- Dice Score
- IoU Score

## Current Status

The coding pipeline of the paper has been implemented.

Currently I have used synthetic CT data for the implementation because the original CARE/WORD dataset is not included in my project.

Full model training and reproduction of the paper results have not been performed yet.
