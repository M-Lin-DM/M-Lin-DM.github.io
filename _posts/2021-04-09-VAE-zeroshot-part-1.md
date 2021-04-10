--- 
title: "Zero-shot Learning with VAEs part 1/3: Building the dataset"
date: 2021-04-09
# layout: "single"
permalink: /VAE_zeroshot_1/
categories:
    - Computer Vision
    - Deep Learning
excerpt: "Introduction and how I construct a known latent space training dataset"
tagline: "Constructing a training set corresponding to a known latent space"
header:
  overlay_image: /images/VAE_zeroshot/banner_2.png
mathjax: "true"
toc: true
toc_label: "Contents"
---

**In Brief**
- I perform two experiments in that involve training a Convolutional variational autoencoder (VAE) on a synthetic image dataset drawn from a known, 2D latent space.
- **Experiment 1 | Parameter space transects (part 2/3):** A model's ability to synthesize images that it received no training data on can be considered **zero-shot learning** in the context of deep generative modeling. I first remove a region of points from the center of a synthetic image dataset. Then I use the trained VAE to sample images from the removed region. **Result:** The model reconstructs images from the removed region fairly well in only a few cases. Highlights the difficulty of interpolating to new regions of the latent space.
- **Experiment 2 | A Loss function for reducing covariance between latent variables (part 3/3):** I train the same VAE on the full image dataset. Here I develop and test a modified loss function designed to minimize covariance between variables in the latent space. The purpose of this loss function is to encourage "disentanglement" of the latent variables, such that each variable encodes a qualitatively orthogonal spectrum of image features. **Result:** The intrinsic two-dimensionality of the training data was confirmed by examining its distribution in a 3D latent space (output of the encoder network). The loss function did not force disentanglement of the latent variables.
- **In this post (part 1/3):** I show how we construct the image dataset with a known latent space.
{: .notice--info}

# Introduction
Within deep learning, *deep generative modeling* deals with models that can generate new images, sound, text, or other rich forms data that would typically be used as the inputs to deep neural nets. Some of the most famous examples in computer vision include generation of never-before-seen human faces using GANs, "deep dream" images, and the transfer of the artistic style of a painting to a photo (style transfer). In this 3-part series I'm going to focus on a type of generative model called variational autoencoders (VAE). I'll cover experiments where we test VAEs' abilities to perform "zero-shot learning," explained later. An in-depth explanation of VAEs is beyond the scope of this post but I'll mention some of the key points. See these sources which give great intros on VAEs: [Variational autoencoders](https://www.jeremyjordan.me/variational-autoencoders/), [Intuitively Understanding Variational Autoencoders](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf), [Understanding VAEs](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73), [Generating new faces with Variational Autoencoders](https://towardsdatascience.com/generating-new-faces-with-variational-autoencoders-d13cfcb5f0a8)

![](/images/VAE_zeroshot/VAE_schematic.jpg)
*Fig. Schematic of a basic VAE with all densely connected layers. A vector of means and log(variance) are produced as a two separate heads at the end of the encoder module. The image $$x^k$$ is mapped to latent vector $$z^k$$ by sampling from gaussian distributions parametrized by the means and log(variances).*


VAEs have some key advantages that make both their training and interpretation easier than in more complex models such as GANs. Like vanilla autoencoders, VAEs can train in a totally unsupervised regime, where the network is tasked with taking in an input image and reconstructing it as output. The data's ground truth labels are the inputs. VAEs are also capable, by design, of learning a continuous *latent space,* from which new samples (images) can be drawn. The latent space refers to the set of underlying variables that control high-level features in the set of objects comprising the dataset. In mapping human faces to a latent space, for example, perhaps there are axes which directly affect masculinity/femininity of the face. In terms of VAE architecture, the *latent vector* $$\mathbf{z}$$ refers to the vector at the middle of the bottleneck (see fig). In contrast to autoencoders, VAEs sample this vector probabalistically by drawing each vector element $$z_i$$ from a normal distribution with mean $$\mu_i$$ and standard deviation $$\sigma_i$$. We can thus think of an image as being mapped to a probability distribution in latent space. And the shape of this distribution is a multivariate gaussian (ellipsoid-like). $$\mu_i$$ and $$\sigma_i$$ are themselves output by two separate 'heads' of the encoder network. By effectively adding noise to the location of the latent vector, we encourage the VAE to produce a smooth latent space, where small changes in the latent vector lead to small changes in the image generated by the decoder. In addition to the image reconstruction loss, VAE's also include a Kullback Leibler divergence loss which forces latent vectors to cluster near the origen. This force helps prevent gaps in the distribution of latent vectors and therefore allows better interpolation between two images (see [Variational autoencoders](https://www.jeremyjordan.me/variational-autoencoders/)). In addition to allowing smooth interpolation between images, the latent space learned by VAEs has even been shown to support "attribute vectors" and arithmetic using such vectors (see [Latent Variable Modeling for Generative Concept Representations and Deep Generative Models](https://www.semanticscholar.org/paper/Latent-Variable-Modeling-for-Generative-Concept-and-Chang/8e629e93a525797ce7f5e704c0dda8f42029451b)).

<figure>
	<a href="/images/VAE_zeroshot/glasses2.jpg"><img src="/images/VAE_zeroshot/glasses2.jpg"></a>
	<figcaption>Slide borrowed from https://www.slideshare.net/andersonljason/variational-autoencoders-for-image-generation.</figcaption>
</figure>


## Research goal | Experiment 1, Parameter space transects
Given the ability of VAEs to interpolate between images, in my first experiment I asked whether a VAE is capable of true **zero-shot learning**, in the sense of generating an image from a part of the latent space it has never been trained on. To make an analogy using the idea of attribute vectors and image interpolation, would it be possible to generate an image of a dog wearing sunglasses, if the model saw many images of humans in sunglasses and dogs without sunglasses during training? In order to test this idea concretely, it helps to have a dataset drawn from a *known,* low-dimensional latent space, so that we can control which parts of the space we train the VAE on. Then, we can ask the VAE to sample images directly from a part of the space it has never seen. Finally we can compare its predictions to known ground truth images. 

To achieve this, I generated a training set of smoothly varying images and then removed a portion of them before training. Each image is a plot of a polar function parametrized by two parameters, $$(a,b)$$. In part two I will generate a "transect" across this parameter space, creating one image per point along the transect. The transect will cross directly over the removed region of the parameter space. Finally, I'll feed these transect images to the VAE and observe their reconstructions.

# Generating the dataset
I wrote matlab code `Generate_latent_space.m` that can be found in this [repository](https://github.com/M-Lin-DM/VAE-zeroshot-learning) to generate and save the images. Each image plots a polar function defined by

$$R(\theta, a, b) = b cos(5 \theta) + a$$

where $$\theta \in [-\pi, \pi]$$ and $$(a,b)$$ are fixed and define the point in the parameter space that corresponds to one image. $$a \in [0.5, 2]$$ controls the radius of the circle and $$b \in [0, 2]$$ controls the amplitude of oscillations superimposed on the circle. In the figure below I varied $$(a,b)$$ over a 10x10 grid in parameter space.


![](/images/VAE_zeroshot/parameter_space_grid.png)
*Fig. Grid sampling of images over a 10x10 lattice in the parameter space. The x-axis corresponds to varying $$a$$ and the y-axis corresponds to $$b$$.*

To generate the full dataset I sampled 5000 points uniformly at random over the range of the parameter space. I then chose a circular region of radius $$r=0.3$$ centered at $$(a=1.25, b=1)$$, and I removed all points inside the region from the dataset. 

```matlab
Npoints = 5000;
data_full = [1.5*rand(Npoints,1)+0.5, 2*rand(Npoints,1)]% [a b]

puncture_location = [1.25 1];
puncture_radius = 0.3;

dists = pdist2(data_full, puncture_location);
data_punctured = data_full(dists>puncture_radius,:);
```

<figure>
	<a href="/images/VAE_zeroshot/Parameter space data_punctured_1p25-1_R0p3.png"><img src="/images/VAE_zeroshot/Parameter space data_punctured_1p25-1_R0p3.png"></a>
	<figcaption>Representation of all training images as points in the parameter space of the polar function. Points inside the hole region are not provided as training data to the model.</figcaption>
</figure>

Finally I plot each curve and save it to the training set folder. The latter 500 of the 5000 images were moved to a testing dataset folder later. I did some additional image processing steps using Fiji/Imagej, such as inverting the images, that are not important. The `.mat` file containing the matlab workspace is also included in my repository if you're interested.

```matlab
R2 = @(th, a, b) b*cos(5*theta) + a;
m=4;
for j = 1:length(data_punctured)

    [x,y]=pol2cart(theta, R2(theta, data_punctured(j,1), data_punctured(j,2)));
    h=plot(x,y,'k','linewidth', 2)
    axis([-m m -m m])
    axis equal
    axis off
    h=gcf
    saveas(h, ['D:\Datasets\VAE_zeroshot\data_punctured_1p25-1_R0p3\img_' sprintf('%04d',j) '.png'])
    close(h)

end
```
## Sample of training images
![](/images/VAE_zeroshot/trainsamp_rs.jpg)
*Fig. Sample of processed training images*