--- 
title: "Zero-shot Learning with VAEs part 3/3: A Loss function for reducing latent variable covariance"
date: 2021-04-12
# layout: "single"
permalink: /VAE_zeroshot_3/
categories:
    - Computer Vision
    - Deep Learning
excerpt: "Changing the VAE loss function to encourage disentanglement of latent variables"
tagline: "Can a modified loss function encourage disentanglement of latent variables?"
header:
  overlay_image: /images/VAE_zeroshot/banner_2.png
mathjax: "true"
toc: true
toc_label: "Contents"
---

- ***Notice:*** Please see [part 1](/VAE_zeroshot_1/) and [part 2](/VAE_zeroshot_2/) for an introduction to this project, or if you'd like to see how I created the dataset. 
- Repository for this project: [Github](https://github.com/M-Lin-DM/VAE-zeroshot-learning)
- Jupyter notebook used: `Train_VAE_entanglement_loss.ipynb`
{: .notice--warning}

**In Brief:** I first generated a synthetic image dataset by drawing from a known latent space. I train a Variational autoencoder (VAE), equipped with a modified loss function, to reconstruct input images from this dataset. This loss function includes an additional term which penalizes embeddings with high covariance among the latent variables. Its purpose is to encourage so-called "disentanglement" of latent variables, so that each variable might encode a qualitatively distinct high-level image feature. In this post, I'll propose and evaluate this modified loss function, recap the VAE architecture I used, and show my results. **Result:** Due to the intrinsic 2-dimensionality of the dataset, the loss function is less suitable in this case. However, the experiment illustrates how the structure of the training data itself could be harnessed to isolate disentangled latent variables. I'll propose hypotheses on such methods.
{: .notice--success}

# Motivation
[Early Visual Concept Learning with Unsupervised Deep Learning](https://arxiv.org/abs/1606.05579)

# Proposed Loss function
![](/images/VAE_zeroshot/VAE_schematic.jpg)
*Fig. Schematic of a basic VAE with all densely connected layers. A vector of means and log(variance) are produced as a two separate heads at the end of the encoder module. The image $$x^k$$ is mapped to latent vector $$z^k$$ by sampling from gaussian distributions parametrized by the means and log(variances).*

| Symbol     | Meaning                                              |
|------------|------------------------------------------------------|
| $$z_i$$        | component i of latent vector                         |
| $$x_p$$        | value at pixel p of input image                      |
| $$\hat{x_p}$$  | value at pixel p of output image                     |
| $$\mu_i$$       | component i of latent vector (before noise is added) |
| $$\sigma_i^2$$ | component i of variance vector                       |
| $$N$$          | number of samples                                    |
| $$d$$          | dimensionality of latent space                       |
| $$k$$          | sample index                                         |
| $$w$$          | image width in pixels                                |
| $$h$$          | image height in pixels                               |

Covariance matrix element:
<div align="center">$$Cov(\mu_i, \mu_j) = C_{i,j} = \sum_{k=1}^N \frac{(\mu_i^k - \bar{\mu_i})(\mu_j^k - \bar{\mu_j})}{N}$$ </div>
Since the KL loss forces the latent embedding's global mean along each axis towards 0 (due to the $$\mu_i^2$$ term), we can assume the center of mass of the embedding will end up roughly at the origin in $$\mathbb{R^d}$$. That is $$\bar{\mu_i} = 0 \forall i$$. Then
<div align="center">$$C_{i,j} = \sum_{k=1}^N \frac{\mu_i^k \mu_j^k}{N}$$</div>

Binary Crossentropy loss:
<div align="center">$$L_{crossent} = -\frac{1}{wh}\sum_{p=1}^{wh} x_p log(\hat{x_p}) + (1-x_p) log(1-\hat{x_p})$$ </div>
![](/images/VAE_zeroshot/Binary_cross_entropy_loss_perpoint.png)

Kullback Leibler divergence loss:
<div align="center">$$L_{KL} = \frac{1}{d}\sum_i^d \mu_i^2 + \sigma_i^2 - 1 - log(\sigma_i^2)$$ </div>
The term $$\mu_i^2$$ simply penalizes 

Entanglement loss:
<div align="center">$$L_{entg} = \frac{1}{d(d-1)/2}\sum_{i=1}^d\sum_{j=i+1}^d |\mu_i \mu_j|$$ </div>


Traditional VAE loss:
<div align="center">$$L = 0.001L_{KL} + L_{crossent}$$ </div>

Modified VAE loss:
<div align="center">$$L = 0.001L_{KL} + L_{crossent} + 0.0001L_{entg}$$ </div>

![](/images/VAE_zeroshot/new_VAE_loss.png)
*Fig. $$L(\mu_i, \mu_j) = \mu_i^2 + \mu_j^2$$ right $$L(\mu_i, \mu_j) = 0.2(\mu_i^2 + \mu_j^2) + 0.8abs(\mu_i \mu_j)$$*

```python
class Add_VAE_Loss(keras.layers.Layer):
    
    def compute_entanglement_loss_component(self, inputs):
        z_mean = inputs
        C_total = 0 #total 'absolute covariance' for this batch
        #cycle through each pair of latent space dimensions and compute covariance-like value. Sum over all pairs to get total covariance
        for i in range(latent_dim):
            for j in range(i+1, latent_dim):
                C_total += K.mean(K.abs(z_mean[:,i]*z_mean[:,j]), axis=0) # sum over all z_mean vectors in the batch 
        
        return C_total*2/(latent_dim*(latent_dim-1)) #we normalize by the number of elements in the upper triangular part of the cov matrix     
    
    def compute_VAE_loss(self, inputs):
        #reasonable values for coeffs: K.mean(0.001*KL_loss + 1.0*recon_loss + 0.0001*entanglement_loss, axis=0)
        sigma = 0.1
        input_image, z_mean, output_image = inputs
        KL_loss = K.mean(K.square(z_mean) + K.exp(K.log(K.square(sigma))) - K.log(K.square(sigma)) - 1, axis=1)
        entanglement_loss = self.compute_entanglement_loss_component(z_mean)
        recon_loss = keras.metrics.binary_crossentropy(K.flatten(input_image), K.flatten(output_image))
        return K.mean(0.001*KL_loss + 1.0*recon_loss + 0.0001*entanglement_loss, axis=0) #sample weight is provided as an output from the dataset (generator)
              
    def call(self, layer_inputs):
        if not isinstance(layer_inputs, list):
            ValueError('Input must be list of [encoder_input, z_mean, z_decoded]')
        loss = self.compute_VAE_loss(layer_inputs)
        self.add_loss(loss)
        return layer_inputs[2]
```

# VAE architecture and training (recap)
Using the Keras-Tensorflow framework, I tested many variations on the architecture and eventually found a version which produced both excellent reconstructions and an intrinsically 2-dimensional latent vector embedding within a 3D latent space (See Figures below). The table below contains the settings I used for this experiment as well as [part 2](/VAE_zeroshot_2/). For more details on the training and hyperparameters, see part 2. Below I'll only cover the settings that are different from those covered part 2!

| Experiment                                            | dataset           | sample weights | latent space dimensionality | Loss function | batch size | optimizer | constant variance |
|-------------------------------------------------------|-------------------|----------------|-----------------------------|---------------|------------|-----------|-------------------|
| Parameter space transects                             | punctured, N=4510 | yes            | 3                           | standard VAE  | 8          | Adam      | yes               |
| Loss function for reducing latent variable covariance | full, N=5000      | no             | 3                           | modified      | 8          | Adam      | yes               |

<figure>
	<a href="/images/VAE_zeroshot/arch.jpg"><img src="/images/VAE_zeroshot/arch.jpg"></a>
    <figcaption>Fig. Tensorflow's model architecture summary.</figcaption>
</figure>

## Dataset
I used the "full" dataset containing 4499 training images and 501 validation images. This dataset did *not* remove 410 images that lied inside the hole region. Each image was automatically resized to 256 x 256 x 1 by the dataset generator (slightly larger dimensions than the original .png files). The training data was provided using a `tf.data.Dataset`, a generator object.

```python
train_dir = 'D:/Datasets/VAE_zeroshot/data_full/processed/train' #your images must be inside a subfolder in the last folder of datadir
img_dataset_train = image_dataset_from_directory(
    train_dir, shuffle = False, image_size=(256, 256), batch_size=8, color_mode='grayscale', validation_split=None, label_mode=None)
```

## Sample Weights
There were no sample weights used for this experiment.


# Results

## Reconstructions from validation set
![](/images/VAE_zeroshot/datafull_validation_preds.png)
*Fig. sd*

## Latent space embedding
[![video link](/images/VAE_zeroshot/data_full_emb_thumb.png)](https://www.youtube.com/watch?v=At9AI-ztdqg "video"){: .align-center}
Click to view video

