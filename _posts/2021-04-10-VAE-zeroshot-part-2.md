--- 
title: "Zero-shot Learning with VAEs part 2/3: Parameter Space Transects"
date: 2021-04-10
# layout: "single"
permalink: /VAE_zeroshot_2/
categories:
    - Computer Vision
    - Deep Learning
excerpt: "Can a VAE reconstruct images from an unseen region of the latent space?"
tagline: "Walking a path through an unknown region of the latent space"
header:
  overlay_image: /images/VAE_zeroshot/banner_2.png
mathjax: "true"
toc: true
toc_label: "Contents"
---

***Notice:*** Please see [part 1](/VAE_zeroshot_1/) for an introduction to this project, or if you'd like to see how I created the dataset used here. In this part, I'll set up the ''parameter space transects'' experiment, go over the VAE architecture I used, and show my results. 

Respository for this project: [Github](https://github.com/M-Lin-DM/VAE-zeroshot-learning)
Jupyter notebook I used for this project: `Train_VAE_sampleweights_constant_variance.ipynb`

**In Brief:** A model's ability to synthesize images that it received no training data on can be considered **zero-shot learning** in the context of deep generative modeling. I generated a synthetic image dataset by drawing from a known latent space. I removed a region of points from this dataset and trained a VAE on the remaining images. I then asked the VAE to sample images along a line extending across the removed region. **Result:** The model reconstructs images from the removed region fairly well in only a few cases. This highlights the difficulty of interpolating to new regions of the latent space.
{: .notice--info}

# Creating transect images
A "transect" in biology or any field research is a straight line along which measurements are taken. For this experiment, we are going to walk on a straight line across the 2D parameter space that the training images are generated from; this line will traverse a region of the space containing images that the VAE has never been trained on. 

I construct 3 transects that cross the $$(a,b)$$ parameter space. $$a$$ and $$b$$ control the size and shape of the polar function plotted in training images. Each transect crosses directly over the center of the "hole" region, where points were removed from the training set. 

![](/images/VAE_zeroshot/PTransects.png)
*Fig. All training images visualized in the parameter space of the polar function $$R(\theta, a, b) = b cos(5 \theta) + a$$. P1, P2, P3 are transects of 15 equally spaced points crossing the entire parameter space.*

Each transect consists in 15 equally-spaced points in parameter space. In `Generate_Transects.m` I simply plot each polar curve along the transect and save as an image into a folder. This produces 15 images per transect that I will feed to the VAE later.


<figure>
	<a href="/images/VAE_zeroshot/Transects_nopred.jpg"><img src="/images/VAE_zeroshot/Transects_nopred.jpg"></a>
</figure>

# VAE architecture and hyperparameters
Using the Keras-Tensorflow framework, I tested many variations on the architecture and eventually found a version which produced both excellent reconstructions and a 2-dimensional latent vector embedding within a 3D latent space. 
<figure>
	<a href="/images/VAE_zeroshot/arch.jpg"><img src="/images/VAE_zeroshot/arch.jpg"></a>
</figure>

* It turned out that only 3 latent variables were needed to produce good reconstructions. I got all results using batch size=8 and the Adam optimizer.
* One major modification, relative to standard VAEs, was to remove entirely the log-variance producing layer from the encoder. Instead, my encoder only outputs a 3D vector of means $$\mu_i$$, and it samples the latent component $$z^k_i$$ for image $$k$$ by assuming a fixed standard deviation of $$\sigma_i = 0.1$$ for all variables $$i$$. This causes the probability distribution of the latent vectors to be spherical and identicle in size for all images. Since the log-variance layer is a dense layer, the total number of parameters in the network also drops substantially. I found that allowing the model to skip learning how to output log-variances actually sped up training and improved performance. The reason for this may have had to do with using an improper activation function on my log-variance layer. 

```python
z_mean = layers.Dense(latent_dim, activation='linear', name='z_mean', kernel_initializer='RandomNormal',  bias_initializer='zeros')(x)

class Sample_latent_vector(keras.layers.Layer):
    def sample(self, inputs):
        sigma = 0.1
        z_mean = inputs
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + epsilon*sigma
    
    def call(self, inputs):
        return self.sample(inputs)
    
z = Sample_latent_vector()(z_mean)
```

* On the layer `z_mean` producing latent component means $$\mu_i$$, I found that a linear activation was best for ensuring that the encoder mapped the training set to a 2-dimensional, sheet-like, embedding (within the 3D latent space)
* I tried applying sample weights so that "starfish" shapes with shallower waves (lower $$b$$) were weighted more heavily in the loss function. This had little effect on the quality of the reconstruction. The quality was more strongly impacted by other factors such as using different activation functions on the log-variance layer (before it was removed). The figure below colorizes the weight used for each sample.

![](/images/VAE_zeroshot/sample_weights.png)

*Fig. Weighted Training dataset visualized in parameter space. Color indicates the values `sample_weights` applied to each sample in the loss function. Yellow is higher value, purple is low. This weighting scheme causes the loss function to increased by more when errors are made on samples with shallower waves (lower $$b$$)*

* The loss function used is the standard VAE loss: a combination of an image reconstruction loss and a Kullback Leibler divergence loss. In part 3 I will test out a novel loss function.

```python
class Add_VAE_Loss(keras.layers.Layer):
    
    def compute_VAE_loss(self, inputs):
        sigma = 0.1
        input_image, z_mean, output_image, sample_weights = inputs
        KL_loss = K.mean(K.square(z_mean) + K.exp(K.log(K.square(sigma))) - K.log(K.square(sigma)) - 1, axis=1)
        recon_loss = keras.metrics.binary_crossentropy(K.flatten(input_image), K.flatten(output_image))
        return K.mean((8e-3 * KL_loss + 1*recon_loss)*sample_weights*2, axis=1) #sample weight is provided as an output from the dataset (generator)
        
        
    def call(self, layer_inputs):
        if not isinstance(layer_inputs, list):
            ValueError('Input must be list of [input_image, z_mean, output_image]')
        loss = self.compute_VAE_loss(layer_inputs)
        self.add_loss(loss)
        return layer_inputs[2]
```

# Results

## Reconstructions from test set
Below is 
![](/images/VAE_zeroshot/punctured_validation_preds.jpg)

## Latent space embedding
dsf
![](/images/VAE_zeroshot/banner.jpg){: .align-center}

fd
![](/images/VAE_zeroshot/mu_embedding_annotated.jpg){: .align-center}


[![video link](/images/VAE_zeroshot/mu_embedding_thumb_resize.png)](https://www.youtube.com/watch?v=uo8HXx9Ik7k "video"){: .align-center}
df

## Transect reconstructions
dfgg
<figure>
	<a href="/images/VAE_zeroshot/Transects_xk_cap_resize.jpg"><img src="/images/VAE_zeroshot/Transects_xk_cap_resize.jpg"></a>
</figure>
sdfs