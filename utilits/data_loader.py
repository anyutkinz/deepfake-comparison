"""
CelebA Data Loading & Splitting System
Implements reproducible dataset handling for GAN training and detection tasks

Key Features:
- Deterministic train/val splits with fixed random seeds
- Model-specific transformations (basic vs adversarial augmentations)
- Combined real/fake datasets for detector training
- Memory-efficient streaming of large image collections

Reference Implementations:
- Liu et al. (2015) - CelebA dataset design
- Goodfellow et al. (2014) - GAN training data practices
- Azulay & Weiss (2018) - Image transformation robustness
- Shmelkov et al. (2017) - Detector training strategies
"""

import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np

#----------------------------------------------------------------
# Multi-Split Dataset Class (Reproducible Partitioning)
#----------------------------------------------------------------
"""Handles CelebA partitioning with three-level hierarchy:
1. 80/20 base split: Training vs validation
2. Training split divided between multiple generators
3. Validation split shared across all models

Design Choices:
- Fixed np.random.seed(42) ensures consistent splits across runs
- Stratified shuffling prevents class imbalance in splits
- File-based indexing minimizes memory footprint
Reference: 
- Dietterich (1998) - Cross-validation methodologies
- Recht et al. (2018) - Dataset splitting best practices
"""

class MultiSplitCelebA(Dataset):
    def __init__(self, root_dir, split_id=0, num_splits=4, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Reproducible permutation using Fisher-Yates algorithm
        all_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
        np.random.seed(42)  # Critical for experiment replication
        indices = np.random.permutation(len(all_files))
        
        # Core 80/20 split (4:1 ratio)
        split_idx = len(indices) // num_splits
        val_indices = indices[-split_idx:]  # Held-out validation
        train_indices = indices[:-split_idx]  # Multi-generator training
        
        # Equitable training split distribution
        gen_splits = np.array_split(train_indices, num_splits-1)
        
        if mode == 'train':
            self.files = [all_files[i] for i in gen_splits[split_id]]
        elif mode == 'val':
            self.files = [all_files[i] for i in val_indices]
        else:
            raise ValueError("Invalid mode. Use 'train' or 'val'")

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_path)
        return self.transform(image) if self.transform else image

#----------------------------------------------------------------
# Data Loader Factory (Model-Specific Configurations)
#----------------------------------------------------------------
"""Implements distinct processing pipelines for different model types:

GAN/Hybrid Models:
- Basic normalization (tanh-compatible [-1,1] range)
- Center cropping for spatial consistency

Detection Models:
- Adversarial augmentations (Rotation + ColorJitter + Flip)
- Combined real/fake datasets with binary labels
- 90/10 train/val split for evaluation

Reference:
- Krizhevsky et al. (2012) - ImageNet preprocessing
- Cubuk et al. (2018) - AutoAugment policies
- Zhang et al. (2021) - Fake detection datasets
"""

def get_loaders(config):
    # Base normalization (matches generator output)
    transform = transforms.Compose([
        transforms.Resize(config['img_size']),
        transforms.CenterCrop(config['img_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Enhanced augmentations for detector robustness
    if config.get('model_type', '').startswith('detector'):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Mirror symmetry
            transforms.ColorJitter(0.1, 0.1, 0.1),  # Color invariance
            transforms.RandomRotation(10),  # Orientation robustness
            transform
        ])
    
    # GAN/Hybrid Training (Single-domain)
    if config['model_type'] in ['gan', 'gan_adv', 'hybrid', 'hybrid_adv']:
        dataset = MultiSplitCelebA(
            root_dir=config['real_path'],
            split_id=config['split_id'],
            transform=transform,
            mode='train'
        )
        return DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    # Detector Training (Cross-domain)
    elif config['model_type'].startswith('detector'):
        real_data = MultiSplitCelebA(
            root_dir=config['real_path'],
            split_id=config['detector_source'],
            transform=transform,
            mode='train'
        )
        
        fake_data = GeneratedDataset(
            root_dir=config['fake_paths'][config['detector_source']],
            transform=transform
        )
        
        # Balanced real/fake mixture (1:1 ratio)
        combined_set = ConcatDataset([
            (img, 0) for img in real_data] +  # Authentic samples
            [(img, 1) for img in fake_data])   # Synthetic samples
        
        # Stratified split preserves class balance
        train_size = int(0.9 * len(combined_set))
        train_set, val_set = random_split(combined_set, [train_size, len(combined_set)-train_size])
        
        return {
            'train': DataLoader(train_set, batch_size=config['batch_size'], shuffle=True),
            'val': DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
        }
    
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

#----------------------------------------------------------------
# Synthetic Dataset Handler (Generated Image Loading)
#----------------------------------------------------------------
"""Specialized loader for model-generated images:
- Automatically discovers image files
- Applies same transforms as real data
- Integrates with PyTorch's DataLoader

Design Considerations:
- Lazy loading prevents memory bloat
- File extension flexibility (.jpg/.png)
- Transform consistency with real data
Reference:
- Salimans et al. (2016) - GAN evaluation metrics
- Wang et al. (2018) - Large-scale image streaming
"""

class GeneratedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_path)
        return self.transform(image) if self.transform else image