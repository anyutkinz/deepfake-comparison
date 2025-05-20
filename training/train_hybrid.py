"""
Hybrid VAE-GAN-Diffusion Training Tutorial
This script implements a hybrid architecture combining:
- Pretrained VAE for latent space manipulation
- GAN framework with TTUR (Two Time-Scale Update Rule)
- Diffusion process for gradual corruption
- Gradient penalty for Lipschitz constraint

Reference implementations from:
- VAE: https://arxiv.org/abs/1312.6114
- Diffusion Models: https://arxiv.org/abs/2006.11239
- TTUR: https://arxiv.org/abs/1706.08500
- WGAN-GP: https://arxiv.org/abs/1704.00028
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utilits.data_loader import MultiSplitCelebA

#----------------------------------------------------------------
# Configuration (Balancing Hybrid Components)
#----------------------------------------------------------------
"""Hyperparameters follow best practices from:
- WGAN-GP: Gulrajani et al. (2017) - gradient penalty coefficient 10
- TTUR: Heusel et al. (2017) - 2:1 discriminator:generator learning rate ratio
- Diffusion: Ho et al. (2020) - linear noise schedule with 200 steps
Design choices:
- Frozen pretrained VAE prevents latent space distortion during GAN training
- Spectral normalization stabilizes GAN training (Miyato et al., 2018)
- Linear diffusion schedule balances corruption speed and training stability
"""

BATCH_SIZE = 64        # Optimized for GPU memory and GAN stability
LATENT_DIM = 256       # Matches pretrained VAE's latent space
EPOCHS = 500           # Longer training for GAN convergence
PATIENCE = 50          # Conservative early stopping for hybrid models
MIN_DELTA = 0.1        # Minimum validation improvement threshold
LEARNING_RATE = 1e-4   # Base LR for TTUR scaling
SPLIT_ID = 2           # Training split index
MAX_DIFFUSION_STEPS = 200  # Full diffusion process length
GRADIENT_PENALTY_COEFF = 10  # WGAN-GP recommendation

config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_path': '/content/drive/MyDrive/deepfake-comparison/data/celeba/img_align_celeba',
    'vae_path': '/content/drive/MyDrive/deepfake-comparison/models/vae/vae_best.pth',
    'models_dir': '/content/drive/MyDrive/deepfake-comparison/models/hybrid',
    'samples_dir': '/content/drive/MyDrive/deepfake-comparison/results/samples/hybrid',
    'split_id': SPLIT_ID,
    'latent_dim': LATENT_DIM,
    'batch_size': BATCH_SIZE,
    'lr': LEARNING_RATE,
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'min_delta': MIN_DELTA
}

#----------------------------------------------------------------
#      Diffusion Scheduler (Linear Noise Schedule)
#----------------------------------------------------------------
"""Implements linear noise schedule for diffusion process
Key features:
- Linearly increasing beta values control noise addition rate
- Cumulative product of alphas tracks total signal preservation
- Closed-form sampling using reparameterization trick
Reference: 
Ho et al. (2020) - Denoising Diffusion Probabilistic Models
Equation: q(x_t|x_0) = N(x_t; sqrt(ᾱ_t)x_0, (1-ᾱ_t)I)"""

class DiffusionScheduler:
    def __init__(self, steps=200, beta_start=0.0001, beta_end=0.01, device='cpu'):
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def apply_diffusion(self, x, t):
        sqrt_alpha = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - self.alpha_cumprod[t]).view(-1, 1)
        noise = torch.randn_like(x)
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise, noise

#----------------------------------------------------------------
#      Architecture Modules (VAE-GAN-Diffusion Integration)
#----------------------------------------------------------------
"""VAE Component (Frozen Weights)
- Pretrained encoder-decoder architecture
- Provides structured latent space for GAN/Diffusion
- No gradient updates to preserve semantic features
Reference: Kingma & Welling (2014)"""

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # Encoder: 64x64x3 -> 8x8x256 -> latent_dim*2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # Downsample to 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim * 2)  # μ and logvar
        )
        
        # Decoder: latent_dim -> 8x8x256 -> 64x64x3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 64x64
            nn.Tanh())  # Match input normalization

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

"""Generator Network (Latent Space Manipulator)
- Maps noise vectors to VAE's latent space
- Spectral normalization prevents mode collapse
- Self-attention captures long-range dependencies
Reference:
- Miyato et al. (2018) - Spectral Normalization
- Zhang et al. (2019) - Self-Attention GAN"""

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Linear(100, 1024)),  # Input noise: 100-dim
            nn.LeakyReLU(0.2),  # Smooth gradient flow
            SelfAttentionBlock(1024),  # Global feature relationships
  
            spectral_norm(nn.Linear(1024, 2048)),
            nn.LeakyReLU(0.2),
            SelfAttentionBlock(2048),

            spectral_norm(nn.Linear(2048, latent_dim)),  # Output matches VAE latent
        )
        
    def forward(self, x):
        return self.main(x)
        
"""Self-Attention Block (Channel-Wise Attention)
- Computes attention weights for feature refinement
- Sigmoid activation produces [0,1] gating values
- Residual connection preserves original features"""
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            spectral_norm(nn.Linear(dim, dim//8)),  # Bottleneck
            nn.ReLU(),
            
            spectral_norm(nn.Linear(dim//8, dim)),  # Feature scaling
            nn.Sigmoid()  # Attention gates
        )
        
    def forward(self, x):
        attn_map = self.attn(x)
        return x * attn_map + x  # Feature recalibration

"""Discriminator Network (Multi-Scale Feature Extraction)
- Progressive downsampling with spectral normalization
- Adaptive pooling before final layer for resolution invariance
- Leaky ReLU prevents dead neurons
Reference:
- Gulrajani et al. (2017) - WGAN-GP architecture
- Karras et al. (2018) - Progressive Growing"""

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pyramid = nn.ModuleList([
            nn.Sequential(  # 64x64 -> 32x32
                spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(  # 32x32 -> 16x16
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(  # 16x16 -> 8x8
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.LeakyReLU(0.2)
            )
        ])
        self.final = spectral_norm(nn.Linear(256, 1))  # Final validity score

    def forward(self, x):
        for layer in self.pyramid:
            x = layer(x)
        return self.final(x.mean([-2, -1]))  # Global average pooling

#----------------------------------------------------------------
#      Hybrid Model Core (Integration Logic)
#----------------------------------------------------------------
"""Combines components through three-stage process:
1. Generator produces latent vectors from noise
2. Diffusion process corrupts latents over time steps
3. VAE decodes diffused latents into images
4. Discriminator evaluates image realism
Key innovations:
- Diffusion in latent space reduces computational complexity
- Frozen VAE enables stable multi-component training
- TTUR balances generator/discriminator learning speeds
Reference: 
- Xiao et al. (2022) - Diffusion in Latent Spaces
- Jolicoeur-Martineau et al. (2021) - Diffusion-GAN Hybrids"""

class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load pretrained VAE (frozen)
        self.vae = VAE(config['latent_dim']).to(config['device'])
        self.vae.load_state_dict(torch.load(config['vae_path']))
        for p in self.vae.parameters():
            p.requires_grad = False
            
        # Trainable components
        self.gen = Generator(config['latent_dim']).to(config['device'])
        self.disc = Discriminator().to(config['device'])
        self.diffusion = DiffusionScheduler(steps=MAX_DIFFUSION_STEPS, device=config['device'])

#----------------------------------------------------------------
#    Data Loading & Splitting (GAN-Specific Augmentations)
#----------------------------------------------------------------
"""Augmentation strategy follows modern GAN practices:
- Random horizontal flips: Dataset symmetry
- Color jitter: Robustness to hue variations
- Random affine: Translation invariance
- Normalization: Tanh-compatible [-1,1] range
Reference:
- Karras et al. (2020) - StyleGAN2 augmentation stack"""

def get_loaders():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = MultiSplitCelebA(
        root_dir=config['data_path'],
        split_id=config['split_id'],
        transform=transform,
        mode='train')

    val_dataset = MultiSplitCelebA(
        root_dir=config['data_path'],
        split_id=3,
        transform=transform,
        mode='val')

    return (
        DataLoader(train_dataset, batch_size=config['batch_size'], 
                  shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_dataset, batch_size=config['batch_size'],
                  shuffle=False, num_workers=4, pin_memory=True))

#----------------------------------------------------------------
#   Training Core & TTUR (Stabilized GAN Optimization)
#----------------------------------------------------------------
"""Implements Two Time-Scale Update Rule:
- Generator: 2e-5 learning rate
- Discriminator: 6e-5 learning rate (3:1 ratio)
Training phases:
1. Discriminator Update:
   - Real/fake sampling with gradient penalty
   - Lipschitz constraint enforcement
2. Generator Update:
   - Latent diffusion process
   - Adversarial loss through discriminator
Reference:
- Heusel et al. (2017) - TTUR convergence proof
- Kodali et al. (2017) - On Convergence in GANs"""

def train():
    os.makedirs(config['models_dir'], exist_ok=True)
    os.makedirs(config['samples_dir'], exist_ok=True)
    
    train_loader, val_loader = get_loaders()
    hybrid = HybridModel(config).to(config['device'])
    early_stopper = EarlyStopping(config['patience'], config['min_delta'])
    
    # TTUR optimizers with 3:1 LR ratio
    optimizerG = optim.Adam(hybrid.gen.parameters(), 
                           lr=2e-5, betas=(0.5, 0.999))  # Slower updates
    optimizerD = optim.Adam(hybrid.disc.parameters(),
                           lr=6e-5,  # Faster discriminator learning
                           betas=(0.5, 0.999))
    
    best_loss = float('inf')
    fixed_noise = torch.randn(64, 100, device=config['device'])

    for epoch in range(config['epochs']):
        hybrid.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        
        for batch_idx, real_imgs in enumerate(train_loader):
            real_imgs = real_imgs.to(config['device'])
            
            # Train Discriminator
            optimizerD.zero_grad()
            
            with torch.no_grad():
                # Generate diffused latents
                z = torch.randn(real_imgs.size(0), 100).to(config['device'])
                clean_z = hybrid.gen(z)
                t = torch.randint(0, MAX_DIFFUSION_STEPS, (real_imgs.size(0),))
                diffused_z, _ = hybrid.diffusion.apply_diffusion(clean_z, t)
                fake_imgs = hybrid.vae.decoder(diffused_z).detach()
            
            # WGAN-GP regularization
            gp = compute_gradient_penalty(hybrid.disc, real_imgs, fake_imgs)
            
            real_validity = hybrid.disc(real_imgs)
            fake_validity = hybrid.disc(fake_imgs.detach())
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + GRADIENT_PENALTY_COEFF * gp
            d_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(hybrid.gen.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(hybrid.disc.parameters(), 1.0)

            optimizerD.step()
            torch.nn.utils.clip_grad_norm_(hybrid.disc.parameters(), 1.0)
            epoch_d_loss += d_loss.item()

            # Train Generator
            optimizerG.zero_grad()
            
            z = torch.randn(real_imgs.size(0), 100).to(config['device'])
            clean_z = hybrid.gen(z)
            t = torch.randint(0, hybrid.diffusion.steps, (real_imgs.size(0),)).to(config['device'])
            diffused_z, _ = hybrid.diffusion.apply_diffusion(clean_z, t)
            fake_imgs = hybrid.vae.decoder(diffused_z)
            
            g_loss = -torch.mean(hybrid.disc(fake_imgs))
            g_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(hybrid.gen.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(hybrid.disc.parameters(), 1.0)
            
            optimizerG.step()
            epoch_g_loss += g_loss.item()
            
            if batch_idx % 50 == 0:
                print(f"[Epoch {epoch+1}/{config['epochs']}] [Batch {batch_idx}/{len(train_loader)}] "
                      f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

        # Added validation monitoring
        hybrid.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_imgs in val_loader:
                val_imgs = val_imgs.to(config['device'])
                z = torch.randn(val_imgs.size(0), 100).to(config['device'])
                clean_z = hybrid.gen(z)
                t = torch.full((val_imgs.size(0),), 50, device=config['device'])
                diffused_z, _ = hybrid.diffusion.apply_diffusion(clean_z, t)
                fake_imgs = hybrid.vae.decoder(diffused_z)
                
                fake_val = hybrid.disc(fake_imgs)
                val_loss += torch.mean(fake_val).item()
        
        avg_val_loss = val_loss/len(val_loader)

        print(f"Epoch {epoch+1} | "
              f"Train D Loss: {epoch_d_loss/len(train_loader):.4f} | "
              f"G Loss: {epoch_g_loss/len(train_loader):.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # Save samples
        hybrid.eval()
        with torch.no_grad():
            z = torch.randn(64, 100, device=config['device'])
            clean_z = hybrid.gen(z)
            t = torch.full((64,), 100, device=config['device'])
            diffused_z, _ = hybrid.diffusion.apply_diffusion(clean_z, t)
            samples = hybrid.vae.decoder(diffused_z)
            save_image(
                samples,
                os.path.join(config['samples_dir'], f"epoch_{epoch+1}.png"),
                nrow=8,
                normalize=True)

        # Early stopping
        current_loss = avg_val_loss
        if current_loss < best_loss - config['min_delta']:
            best_loss = current_loss
            early_stopper.counter = 0
            torch.save(hybrid.gen.state_dict(), 
                      os.path.join(config['models_dir'], 'generator_best.pth'))
            torch.save(hybrid.disc.state_dict(),
                      os.path.join(config['models_dir'], 'discriminator_best.pth'))
        else:
            early_stopper.counter += 1
            
        if early_stopper(current_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training complete! Best loss:", best_loss)

#----------------------------------------------------------------
#      Utility Functions
#----------------------------------------------------------------
"""Early Stopping Mechanism
- Monitors validation loss improvement
- Stops training after patience epochs without improvement
Reference: https://arxiv.org/abs/1710.05468 (Early Stopping)"""
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

"""Gradient Penalty Calculation
- Enforces 1-Lipschitz constraint via interpolated samples
- Critical for WGAN-GP training stability
Reference: https://arxiv.org/abs/1704.00028 (WGAN-GP)"""
def compute_gradient_penalty(D, real_samples, fake_samples):
    device = real_samples.device
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    if gradients is None:
        return torch.tensor(0.0, device=device)
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

if __name__ == "__main__":
    train()