"""
Wasserstein GAN-GP Training Tutorial
This script implements the WGAN-GP architecture with comprehensive stabilization techniques:

Key Components:
- Gradient penalty for Lipschitz constraint enforcement (Gulrajani et al., 2017)
- Instance normalization for generator stability (Ulyanov et al., 2016)
- RMSprop optimizer following original implementation guidelines
- Conservative early stopping based on Wasserstein distance

Reference implementations from:
- WGAN-GP: https://arxiv.org/abs/1704.00028 (Gulrajani et al., 2017)
- Instance Normalization: https://arxiv.org/abs/1607.08022 (Ulyanov et al., 2016)
- DCGAN Framework: https://arxiv.org/abs/1511.06434 (Radford et al., 2015)
"""

import sys
import os
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__), '..')))

from utilits.data_loader import MultiSplitCelebA

#----------------------------------------------------------------
# Configuration (WGAN-GP Paper Settings)
#----------------------------------------------------------------
"""Hyperparameters follow original WGAN-GP recommendations:
- N_CRITIC: 5 discriminator updates per generator update
- LAMBDA_GP: 10 gradient penalty coefficient (Eq. 3 in Gulrajani et al.)
- LEARNING_RATE: 5e-5 for RMSprop (no momentum as per paper)
- PATIENCE: 50 epochs for early stopping
Design Choices:
- Instance normalization prevents feature magnitude explosion in generator
- No batch normalization in discriminator for stable gradient penalty
- Tanh output matches data normalization to [-1,1] range
- Leaky ReLU (α=0.2) in discriminator prevents sparse gradients
"""

BATCH_SIZE = 64        # Optimal balance between VRAM and batch diversity
IMAGE_SIZE = 64        # Standard resolution for CelebA experiments  
LATENT_DIM = 128       # Dimensionality of input noise vectors
EPOCHS = 500           # Sufficient for convergence with stabilization methods
N_CRITIC = 5           # Discriminator updates per generator update (Algorithm 1)
LAMBDA_GP = 10         # Gradient penalty coefficient (λ in paper)
LEARNING_RATE = 5e-5   # RMSprop learning rate from original implementation
PATIENCE = 50          # Early stopping patience for Wasserstein distance

config = {
    'data_path': '/content/drive/MyDrive/deepfake-comparison/data/celeba/img_align_celeba',
    'samples_dir': '/content/drive/MyDrive/deepfake-comparison/results/samples/wgan_gp',
    'models_dir': '/content/drive/MyDrive/deepfake-comparison/models/wgan_gp',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

#----------------------------------------------------------------
# Model Architectures (Stabilized DCGAN Variant)
#----------------------------------------------------------------
"""Generator Network (Transposed Convolutional Architecture)
- Progressive upsampling: 4x4 → 64x64 through 5 deconv layers
- Instance normalization after each layer except output
- ReLU activations for non-linearity with gradient flow
- Tanh output matches normalized image range
Reference: 
- Radford et al. (2015) - DCGAN architecture guidelines
- Ulyanov et al. (2016) - Instance normalization benefits"""

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

"""Discriminator Network (Critic with Spectral Normalization)
- Progressive downsampling: 64x64 → 4x4 through 4 conv layers
- Spectral normalization enforces Lipschitz constraint
- Leaky ReLU (α=0.2) prevents gradient vanishing
- No final activation for Wasserstein estimates
Reference:
- Miyato et al. (2018) - Spectral Normalization benefits
- Arjovsky et al. (2017) - Wasserstein metric properties"""

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(256, 1, 4, 1, 0))
        )
        
    def forward(self, x):
        return self.main(x)

#----------------------------------------------------------------
# Gradient Penalty Calculation (Lipschitz Constraint)
#----------------------------------------------------------------
"""Computes gradient penalty for interpolated samples
Equation: λ * E[(||∇D(x̂)||₂ - 1)²] where x̂ = εx + (1-ε)G(z)
Implementation Details:
- Random convex combinations of real/fake samples
- Dual backprop for gradient computation (retain_graph=True)
- L2 norm penalty centered at 1 (1-Lipschitz enforcement)
Reference: Algorithm 1 in Gulrajani et al. (2017)"""

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=config['device'])
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return LAMBDA_GP * gradient_penalty

#----------------------------------------------------------------
# Main Training Function (WGAN-GP Optimization)
#----------------------------------------------------------------
"""Training process follows WGAN-GP paper specifications:
1. Critic Phase (N_CRITIC steps):
   - Sample real and generated batches
   - Compute Wasserstein estimate and gradient penalty
   - Update critic with RMSprop (no momentum)
2. Generator Phase:
   - Update generator to minimize -D(G(z))
3. Early Stopping:
   - Track Wasserstein distance (negative critic loss)
   - Patience-based stopping for convergence

Data Augmentation:
- Random horizontal flips (p=0.5) for dataset symmetry
- Color jitter for illumination robustness
- Center cropping to 64x64 resolution
Reference: Section 4 in Gulrajani et al. (2017) - Training Details"""

def main():
    os.makedirs(config['samples_dir'], exist_ok=True)
    os.makedirs(config['models_dir'], exist_ok=True)

    if not os.path.exists(config['data_path']):
        raise FileNotFoundError(f"Dataset not found at {config['data_path']}")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = MultiSplitCelebA(
        root_dir=config['data_path'],
        split_id=1,
        transform=transform,
        mode='train'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2 if config['device'] == 'cuda' else 0,
        pin_memory=True
    )

    netG = Generator().to(config['device'])
    netD = Discriminator().to(config['device'])

    optimizerG = optim.RMSprop(netG.parameters(), lr=LEARNING_RATE)
    optimizerD = optim.RMSprop(netD.parameters(), lr=LEARNING_RATE)

    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=config['device'])

    best_wasserstein = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        netG.train()
        netD.train()
        
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(config['device'])
            
            # Critic training
            for _ in range(N_CRITIC):
                optimizerD.zero_grad()
                noise = torch.randn(real_imgs.size(0), LATENT_DIM, 1, 1, device=config['device'])
                fake_imgs = netG(noise)
                
                real_validity = netD(real_imgs)
                fake_validity = netD(fake_imgs.detach())
                
                gradient_penalty = compute_gradient_penalty(netD, real_imgs.data, fake_imgs.data)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
                d_loss.backward()
                optimizerD.step()
            
            # Generator training
            optimizerG.zero_grad()
            fake_imgs = netG(noise)
            g_loss = -torch.mean(netD(fake_imgs))
            g_loss.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

        # Save samples and check early stopping
        with torch.no_grad():
            fake = netG(fixed_noise).cpu()
            save_image(fake, os.path.join(config['samples_dir'], f"epoch_{epoch+1}.png"), 
                      nrow=8, normalize=True)
        
        current_wasserstein = -d_loss.item()
        if current_wasserstein > best_wasserstein:
            best_wasserstein = current_wasserstein
            patience_counter = 0
            torch.save(netG.state_dict(), os.path.join(config['models_dir'], 'generator_best.pth'))
            torch.save(netD.state_dict(), os.path.join(config['models_dir'], 'discriminator_best.pth'))
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training complete. Models saved to:", config['models_dir'])

if __name__ == "__main__":
    main()