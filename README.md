# **Training VAE, GAN and Hybrid Models on (small) CelebA Dataset**

## `Note:`

***On improvements:***

*This project is not fully completed (according to the author's subjective assessment), however, at this stage it can serve as a good educational resource on the implementation of various generative models. Feel free to send me feedback!*

*In the future, I plan to expand it (see the *Future Improvements* section at the end of this file to learn what exactly I want to add), so you are always welcome to return and observe the progress :)*

***On files:***

Some of the folders are too large to be uploaded to GitHub. If you're interested in the generated images from the training process, my CelebA subset (24K images), and the models, feel free to visit my Google Drive: https://drive.google.com/drive/folders/1-1AAcsYdo19QZUa-SFp7itK73LRC95IE?usp=sharing

## `Introduction:`
Anyone who begins their journey in a rapidly developing field, in our case Deep Learning, may find themselves in a challenging position, specifically overwhelmed by numerous research papers, each of which incrementally improves classical models. I found myself in exactly this situation.

What options do we have in such circumstances? All we can do is meticulously search for "ideal" solutions across various papers and experiment, experiment, experiment...

## `Goal:`
In this repository, you will find my attempts to create stable scripts for various generative models, combining different techniques and design solutions. The main objective is to explore and implement state-of-the-art approaches to image generation while making the implementations accessible and educational. By integrating insights from multiple papers, I aim to create reliable generative models that can be easily understood and modified.

This README file contains only a summary of the scripts, so if you're interested in details - please take the time to look at the scripts themselves. I've tried to leave comprehensive comments inside each script to guide you through the implementation choices and technical considerations.

## `Dataset:`
The [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (Celebrities Attributes) is a large-scale face attributes dataset containing more than 200,000 celebrity images with 40 attribute annotations. It's widely used for training and testing face recognition algorithms, generative models, and various computer vision tasks due to its size, quality, and attribute annotations.

Due to limited resources, I used only a small subset of CelebA consisting of 24,000 images.

### Why this specific amount?
According to several research papers, particularly those focusing on GAN training dynamics, approximately 8,000-10,000 images are needed for proper GAN convergence on tasks of this complexity. Using 24,000 images provides a good balance between having enough training data while still being manageable with limited computational resources.

### Data Splitting Strategy
As shown in the `data_loader.py` script, I implemented a multi-split approach to handle different training scenarios:

1. The dataset is first divided into 80% training and 20% validation sets
2. The training portion is further split into 4 equal parts (for different generator models)
3. Each generator model uses its designated split (identified by `split_id`)
4. This approach ensures that:
   - Different models train on different subsets of data (preventing memorization)
   - We can fairly compare model performance since they train on equal-sized datasets
   - The validation set remains consistent across all models

The `MultiSplitCelebA` class handles these splits with a reproducible random seed (42) to ensure consistency across runs.

Now let's dive into our models!

## `VAE (Variational Autoencoder):`

Variational Autoencoders, introduced by Kingma and Welling in their 2014 paper "Auto-Encoding Variational Bayes," represent a significant advancement in generative models. VAEs combine elements of variational inference and neural networks to create a probabilistic framework for generating new data.

At its core, a VAE consists of:
1. An ,**encoder** network that maps input data to a probability distribution in the latent space
2. A **decoder** network that reconstructs the original input from samples of this distribution

The key innovation is that VAEs learn a continuous latent space where similar inputs are mapped to nearby points, enabling smooth interpolation and generation of new samples.

The VAE is trained by optimizing the Evidence Lower Bound (ELBO):

![Formula](https://latex.codecogs.com/svg.image?$\mathcal{L}(\theta,\phi;x)=\mathbb{E}{q{\phi}(z|x)}[\log&space;p_{\theta}(x|z)]-D_{\text{KL}}(q_{\phi}(z|x)\parallel&space;p(z))$)

Where the first term is the reconstruction loss and the second term is the Kullback-Leibler divergence that regularizes the latent space.

### Implementation Details:

While I tried to maintain the classical VAE architecture, I incorporated several enhancements to improve training stability and generation quality:

- **Cyclical KL Annealing**: Based on Bowman et al. (2016), I implemented a cyclical annealing schedule for the KL divergence term. This prevents the "posterior collapse" problem where the model ignores the latent space. The `CyclicalAnnealer` class gradually increases the weight of the KL term over multiple cycles.

- **Convolutional Architecture**: Rather than using fully connected layers, I implemented a symmetrical convolutional encoder-decoder architecture, which is better suited for image data.

- **Data Augmentation**: To prevent overfitting, I included various augmentations:
  - Random affine transformations (±10 degrees)
  - Gaussian blur with variable sigma
  - Horizontal flips
  - Color jitter

- **Early Stopping**: To prevent overfitting, training stops after 10 epochs without improvement in validation loss.

- **MSE Reconstruction Loss**: Instead of binary cross-entropy, I used Mean Squared Error which works better for the continuous pixel values in this case.

The hyperparameters were carefully tuned:
- Latent dimension of 256 (compromise between reconstruction quality and meaningful latent space)
- Learning rate of 1e-4 (standard for Adam optimizer)
- KL weight of 0.05 (found through experimentation to balance reconstruction vs. regularization)


Looking at the generated images, we can see that the VAE produces relatively coherent face structures. The results demonstrate the characteristic "blurriness" common to VAEs, which stems from the probabilistic nature of the model and the use of MSE loss which tends to average out details.

The facial features are recognizable and consistent across samples, showing that the model has learned meaningful representations. Color tones and general face shapes are captured well, though fine details like hair strands or small facial features lack sharpness. This is expected behavior for VAEs, as they prioritize capturing the overall distribution rather than sharp details.

The model shows good stability in training, with relatively consistent quality across different samples, indicating that the cyclical KL annealing technique was effective in balancing the reconstruction vs. regularization trade-off.

## `GAN (Generative Adversarial Network):`

Generative Adversarial Networks, introduced by Goodfellow et al. in 2014, represent a fundamentally different approach to generative modeling. GANs consist of two neural networks - a generator and a discriminator - engaged in a minimax game:

1. The **generator** creates samples to fool the discriminator
2. The **discriminator** tries to distinguish between real and generated samples

This adversarial training framework is formalized as:

![Formula](https://latex.codecogs.com/svg.image?$$\min_G\max_D&space;V(D,G)=\mathbb{E}_{x\sim&space;p_{\text{data}}}[\log&space;D(x)]&plus;\mathbb{E}_{z\sim&space;p_z}[\log(1-D(G(z)))]$$)

Where *G* is the generator, *D* is the discriminator, *x* represents real data, and *z* is random noise input to the generator.

### Implementation Details:

Referring to the `train_classic_gan.py` script, I implemented a Wasserstein GAN with gradient penalty (WGAN-GP), incorporating several techniques from recent literature:

- **Wasserstein Distance**: Instead of the original GAN formulation, I used the Wasserstein distance (from Arjovsky et al., 2017) which provides more stable gradients during training.

- **Gradient Penalty**: As proposed by Gulrajani et al. (2017), I added a gradient penalty term to enforce the 1-Lipschitz constraint on the critic (discriminator), which greatly improves training stability.

- **Instance Normalization**: Used in the generator instead of batch normalization, as it's been shown to produce better results for image generation tasks (Ulyanov et al., 2016).

- **Spectral Normalization**: Applied to the discriminator's weights to further enforce Lipschitz constraints and stabilize training.

- **RMSprop Optimizer**: As recommended in the WGAN paper, RMSprop works better than Adam for Wasserstein distance optimization.

- **Multiple Critic Updates**: For each generator update, the critic is updated 5 times, ensuring it provides useful feedback to the generator.

This model consumed a considerable amount of my time because the results were initially quite unstable. Common instability patterns included:

- Mode collapse (generating only a single type of face)
- Oscillating losses without convergence
- Extremely noisy or distorted outputs
- Gradient explosion or vanishing gradients

However, as it turned out, a combination of numerous papers, patience, and an intuitive understanding of where to adjust hyperparameters can stabilize even the most unpredictable model.

The techniques mentioned above, particularly the gradient penalty and spectral normalization, were crucial in achieving stability. Additionally, careful tuning of the learning rate (5e-5) and patience-based early stopping helped the model converge to reasonable results.

Here are my unique generated images:



Of course, some of them still look quite "questionable." You could even say they appear somewhat frightening, especially during late-night sessions working on this project. However, the result is genuinely not that bad!

Based on the images, we can observe:

- **Higher diversity**: Compared to the VAE, there's a wider range of facial features, hairstyles, and expressions
- **Better color saturation**: The colors are more vibrant and realistic
- **Sharper details**: Individual facial features are more defined than in VAE outputs
- **Structural inconsistencies**: Some faces show anatomical irregularities or asymmetries
- **Artifacts**: Occasional noise or distortion patterns appear in some samples
- **Identity confusion**: Some images show blending of multiple facial characteristics

The model still falls short of photorealism because:
1. The limited dataset size (only a portion of CelebA)
2. The inherent training instability of GANs
3. The relatively simple architecture compared to state-of-the-art models like StyleGAN2

Nevertheless, the results demonstrate that the implemented stability techniques were effective in producing coherent face images with reasonable diversity.

## `Hybrid Model:`

To make a long story short - this model was completely unsuccessful in the training process. I couldn't achieve stable training, and consequently, there's no possibility of convergence. Nevertheless, I still have some hopes for this model!

The `train_hybrid.py` script implements a sophisticated hybrid architecture combining three powerful generative approaches:

1. **VAE component**: A pretrained VAE provides the latent space manipulation capabilities
   - Uses the encoder-decoder structure from the standard VAE
   - The weights are frozen during hybrid model training
   - Serves as a reconstruction module with perceptual understanding

2. **GAN framework**: Implements adversarial training with several modern techniques
   - Uses Two Time-Scale Update Rule (TTUR) from Heusel et al. (2017) for stable adversarial training
   - Different learning rates for generator (2e-5) and discriminator (6e-5)
   - Includes spectral normalization and self-attention blocks for improved sample quality

3. **Diffusion process**: Introduces a gradual noise schedule in the latent space
   - Based on Ho et al.'s (2020) Denoising Diffusion Probabilistic Models
   - Implements a forward diffusion scheduler with linear beta schedule
   - Corrupts the latent representation with increasing levels of noise
   - Helps smooth the latent space and improve sample diversity

Additional techniques implemented:
- **Self-attention mechanism**: Based on Zhang et al. (2019) to capture long-range dependencies
- **Gradient penalty**: From WGAN-GP for Lipschitz constraint enforcement
- **Early stopping**: With patience-based monitoring of validation loss
- **Adaptive gradient clipping**: To prevent gradient explosion during training

The hybrid approach was inspired by several recent papers exploring combinations of generative paradigms:
- VQ-VAE-2 (Razavi et al., 2019) for hierarchical latent representations
- NVAE (Vahdat & Kautz, 2020) for deep hierarchical VAEs
- Diffusion-GAN (Xiao et al., 2021) for combining diffusion models with GANs

Obviously, this model is the most complex, and therefore requires a more careful approach to hyperparameter adjustments, corrections, and thus a large number of experiments which I, in the circumstances of limited computational resources and budget for continuing to purchase subscriptions to Google Colab Pro, decided to temporarily (!) suspend.

Nevertheless, if we look at the generated images, we can see that even a "poorly" trained model produces not-so-bad results. Although they are not as diverse and a bit blurry, we can still notice that the facial features in this case are most similar to real faces, rather than low-budget horror games, as in the case of the GAN.

## `Future Improvements:`

### Hybrid Model
For the reasons described above, I dare to assume that with more time dedicated to experimentation, I could achieve good results from this model. Stay tuned or if you have any suggestions - feel free to share your opinion on implementation!

### FID / Other Evaluation
At this stage, I judged the quality of the model (stability of training and convergence) based on the loss values and quality of generated images, which of course is not such a reliable source. What are the alternatives?

Several quantitative evaluation metrics could be implemented:

1. **Fréchet Inception Distance (FID)**: Measures the similarity between generated and real image distributions by comparing features extracted by an Inception network.

2. **Inception Score (IS)**: Evaluates both the quality and diversity of generated images, though it has known limitations when applied to face generation.

3. **Perceptual Path Length (PPL)**: Measures the smoothness of the latent space, indicating how well the model generalizes.

Implementing these metrics would provide more objective comparisons between the different models and track improvements over iterations.

### Deepfake Detector Training
In the future, this project could be extended by shifting the focus from generative models to detector training, specifically measuring how detectors trained in an adversarial loop with different models differ in their effectiveness. My hypothesis is that the hybrid model might combine the speed of GANs and the quality of Diffusion Models, making it the most advantageous option. Using an adversarial loop, we could improve the quality of not only the model itself but also the detector accordingly.

Such enhancement could become a kind of benchmark, but I hope to tell you more about this later.


