"""
Variational Autoencoder Training with Cyclical KL Annealing
This script implements a VAE with:
- Convolutional encoder/decoder symmetry
- Cyclical KL annealing for stable training (Bowman et al., 2016)
- Mean Squared Error (MSE) reconstruction loss
Reference implementations from:
- VAE: Kingma & Welling (2014) - https://arxiv.org/abs/1312.6114
- KL Annealing: Bowman et al. (2016) - https://arxiv.org/abs/1511.06349
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utilits.data_loader import MultiSplitCelebA

#----------------------------------------------------------------
# Cyclic KL annealing (Bowman et al., 2016 implementation)
#----------------------------------------------------------------

class CyclicalAnnealer:
    """Implements cyclical KL annealing from NAACL 2019 paper
    Args:
        total_steps: Total training steps (epochs * batches_per_epoch)
        n_cycles: Number of annealing cycles
        ratio: Portion of cycle spent ramping up (0.5 = 50% ramp up)
    """
    def __init__(self, total_steps, n_cycles=4, ratio=0.5):
        self.step = 0
        self.cycle_length = total_steps // n_cycles  # Steps per cycle
        self.ramp_up = int(self.cycle_length * ratio)  # Ramp-up phase duration
        self.n_cycles = n_cycles
        
    def get_beta(self):
        """Calculate current annealing factor (β) using triangular schedule"""
        cycle = self.step // self.cycle_length  # Current cycle index
        cycle_step = self.step % self.cycle_length  # Position within cycle
        
        # Linear ramp-up within cycle
        if cycle_step < self.ramp_up:
            beta = cycle_step / self.ramp_up  # [0,1)
        else:
            beta = 1.0  # Full KL weight
            
        self.step += 1
        return min(1.0, beta * (cycle + 1) / self.n_cycles)  # Scale by cycle number

#----------------------------------------------------------------
# Configuration (Balancing Reconstruction and KL Loss)
#----------------------------------------------------------------
# Parameters follow VAE best practices from Kingma & Welling (2014)
# - KL_WEIGHT: Final weight after annealing to prevent posterior collapse
# - LATENT_DIM: Trade-off between reconstruction quality and latent space structure
# - CYCLICAL_ANNEALING: Reduces KL divergence early training instability

BATCH_SIZE = 64        # Optimized for GPU memory and batch normalization
IMAGE_SIZE = 64         # Standard CelebA preprocessing size
LATENT_DIM = 256        # Compromise between complexity and disentanglement
EPOCHS = 100             # Sufficient for convergence with early stopping
LEARNING_RATE = 1e-4    # Default Adam LR from Kingma & Ba (2015)
KL_WEIGHT = 0.05         # Final KL weight after annealing
PATIENCE = 10            # Conservative early stopping for small datasets
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path configurations (preserve project structure)
DATA_PATH = '/content/drive/MyDrive/deepfake-comparison/data/celeba/img_align_celeba'
SPLIT_ID = 0            # Training split index
SAMPLES_DIR = '/content/drive/MyDrive/deepfake-comparison/results/samples/vae/recon'  # Reconstruction samples
MODELS_DIR = '/content/drive/MyDrive/deepfake-comparison/models/vae'  # Model checkpoints
GENERATED_DIR = SAMPLES_DIR = '/content/drive/MyDrive/deepfake-comparison/results/samples/vae/generated'

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

#----------------------------------------------------------------
# Data Loading with Augmentations (Prevent Overfitting)
#----------------------------------------------------------------
# Augmentations follow modern VAE practices:
# - RandomAffine: Simulate viewpoint variations
# - GaussianBlur: Improve robustness to input noise
# - ColorJitter: Domain randomization for generalization

transform = transforms.Compose([
    transforms.RandomAffine(degrees=10),  # Random viewpoint variations
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Robustness to blur
    transforms.RandomHorizontalFlip(p=0.5),  # Dataset symmetry augmentation
    transforms.ColorJitter(0.1, 0.1, 0.1),  # Mild color variations
    transforms.Resize(IMAGE_SIZE),  # Standardize input size
    transforms.CenterCrop(IMAGE_SIZE),  # Consistent composition
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Tanh-compatible
])

# Training dataset (split 0)
train_dataset = MultiSplitCelebA(
    root_dir=DATA_PATH,
    split_id=SPLIT_ID,
    transform=transform,
    mode='train'
)

# Validation dataset (split 1 for true holdout)
val_dataset = MultiSplitCelebA(
    root_dir=DATA_PATH,
    split_id=3,  # Different split for validation
    transform=transform,
    mode='val'
)

# DataLoader configurations
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # Prevent order memorization
    num_workers=2,  # Disable for Colab compatibility
    pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # Preserve validation order
    num_workers=2,
    pin_memory=True
)

#----------------------------------------------------------------
# VAE Architecture (Kingma & Welling, 2014 Base)
#----------------------------------------------------------------
# Symmetrical Conv2d/ConvTranspose2d design with:
# - ReLU activations except output layer
# - Tanh output to match input normalization
# - Reparameterization trick for differentiable sampling

class VAE(nn.Module):
    """Vanilla VAE with convolutional encoder/decoder (Kingma & Welling, 2014)
    Design choices:
    - Symmetrical architecture for encoder/decoder
    - ReLU activations for non-linearity
    - Tanh output for normalized pixel values
    Reference:
    Kingma, D.P. & Welling, M. (2014). Auto-Encoding Variational Bayes. arXiv:1312.6114.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Encoder (q(z|x)): 64x64x3 -> 8x8x256 -> latent_dim*2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 64x64 -> 32x32
            nn.ReLU(),  # Non-linearity
            nn.Conv2d(64, 128, 4, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten(),  # 8x8x256 -> 16384
            nn.Linear(256 * 8 * 8, latent_dim * 2)  # μ and logσ²
        )
        
        # Decoder (p(x|z)): latent_dim -> 8x8x256 -> 64x64x3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),  # Project to decoder input
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),  # Reshape to spatial features
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 32x32 -> 64x64
            nn.Tanh()  # Match input normalization
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick (differentiable sampling)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Noise from N(0,1)
        return mu + eps * std  # Scale and shift

    def forward(self, x):
        """Full forward pass with reconstruction and KL components"""
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)  # Split encoder output
        z = self.reparameterize(mu, logvar)  # Sampled latent
        x_recon = self.decoder(z)  # Reconstructed output
        return x_recon, mu, logvar

#----------------------------------------------------------------
# Training Setup (Optimization and Tracking)
#----------------------------------------------------------------
# Uses Adam optimizer as per original VAE paper
# Cyclical annealing follows Bowman et al. (2016) implementation

vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)  # Default β1/β2
total_steps = EPOCHS * len(train_dataloader)  # Total training iterations
annealer = CyclicalAnnealer(total_steps, n_cycles=4, ratio=0.3)  # KL scheduler
best_val_loss = float('inf')
counter = 0  # Early stopping counter

fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)

#----------------------------------------------------------------
# Training Loop with Early Stopping and Sample Saving
#----------------------------------------------------------------
# Implements:
# - Cyclical KL annealing during training
# - Full KL weight for validation
# - Early stopping based on validation loss

for epoch in range(EPOCHS):
    # Training phase (weight updates)
    vae.train()
    total_loss = 0
    for batch_idx, real_imgs in enumerate(train_dataloader):
        real_imgs = real_imgs.to(DEVICE)
        optimizer.zero_grad()
        
        # Forward pass
        recon_imgs, mu, logvar = vae(real_imgs)
        
        # Loss calculation
        current_beta = annealer.get_beta()  # Get annealing factor
        effective_kl_weight = current_beta * KL_WEIGHT  # Scale KL term
        
        recon_loss = nn.MSELoss()(recon_imgs, real_imgs)  # L2 reconstruction
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # KL(q||p)
        loss = recon_loss + effective_kl_weight * kl_loss  # ELBO
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    vae.eval()
    with torch.no_grad():
            # Генерация из фиксированного шума
        generated = vae.decoder(fixed_noise)
        save_image(
            generated,
            os.path.join(GENERATED_DIR, f"epoch_{epoch+1}.png"),
            nrow=8,
            normalize=True
        )
            
    # Validation phase (no gradients)
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_imgs in val_dataloader:
            val_imgs = val_imgs.to(DEVICE)
            recon_val, mu_val, logvar_val = vae(val_imgs)
            
            # Validation loss (full KL weight)
            recon_loss = nn.MSELoss()(recon_val, val_imgs)
            kl_loss = -0.5 * torch.mean(1 + logvar_val - mu_val.pow(2) - logvar_val.exp())
            loss_val = recon_loss + KL_WEIGHT * kl_loss
            
            val_loss += loss_val.item()
    
    val_loss /= len(val_dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {total_loss/len(train_dataloader):.4f} | Val Loss: {val_loss:.4f}")

    # Early stopping and model checkpointing
    if val_loss < best_val_loss:  # Improvement detected
        best_val_loss = val_loss
        counter = 0
        
        # Save best model weights
        torch.save(vae.state_dict(), os.path.join(MODELS_DIR, "vae_best.pth"))
        
        # Save reconstruction samples
        vae.eval()
        with torch.no_grad():
            sample_imgs = next(iter(train_dataloader)).to(DEVICE)
            recon_imgs, _, _ = vae(sample_imgs)
            save_image(
                torch.cat([sample_imgs, recon_imgs], dim=0),  # Original|Reconstructed
                os.path.join(SAMPLES_DIR, f"recon_best.png"),
                nrow=BATCH_SIZE,
                normalize=True
            )
        print("Saved new best model and samples!")
    else:  # No improvement
        counter += 1
        print(f'EarlyStopping counter: {counter}/{PATIENCE}')
        if counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            
            # Final samples before exit
            vae.eval()
            with torch.no_grad():
                sample_imgs = next(iter(train_dataloader)).to(DEVICE)
                recon_imgs, _, _ = vae(sample_imgs)
                save_image(
                    torch.cat([sample_imgs, recon_imgs], dim=0),
                    os.path.join(SAMPLES_DIR, f"recon_final.png"),
                    nrow=BATCH_SIZE,
                    normalize=True
                )
            break

    # Periodic checkpoints (every 5 epochs)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        vae.eval()
        with torch.no_grad():
            sample_imgs = next(iter(train_dataloader)).to(DEVICE)
            recon_imgs, _, _ = vae(sample_imgs)
            save_image(
                torch.cat([sample_imgs, recon_imgs], dim=0),
                os.path.join(SAMPLES_DIR, f"recon_epoch_{epoch+1}.png"),
                nrow=BATCH_SIZE,
                normalize=True
            )
        torch.save(vae.state_dict(), os.path.join(MODELS_DIR, f"vae_epoch_{epoch+1}.pth"))

# Final model save
torch.save(vae.state_dict(), os.path.join(MODELS_DIR, "vae_final.pth"))
print("VAE training complete! Best model saved to:", os.path.join(MODELS_DIR, "vae_best.pth"))