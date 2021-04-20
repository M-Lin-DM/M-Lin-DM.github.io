--- 
title: "Image-Learning-based Texture Synthesis part 2/2: Dataset and algorithm"
date: 2021-04-20
# layout: "single"
permalink: /FCN_Lichen_textures_2/
categories:
    - Computer Vision
    - Deep Learning
excerpt: "Extracting textures from natural images"
tagline: "Extracting textures from natural images"
header:
  overlay_image: /images/FCN_Lichen_textures/banner.png
mathjax: "true"
toc: true
toc_label: "Contents"
---

- ***Notice:*** Please see [part 1](/FCN_Lichen_textures_1/) for an introduction to this project.
- Repository for this project: [Github](https://github.com/M-Lin-DM/VAE-zeroshot-learning)
- Jupyter notebook used: `Train_VAE_sampleweights_constant_variance.ipynb`
{: .notice--warning}

**In Brief** 
- In deep generative modeling, **Texture synthesis** is the process of creating novel texture images. **Texture extraction** involves taking a natural image and producing images with qualitatively similar texture properties. 
- In this project, I develop a novel texture extraction model that uses image-learning-based techniques. This make it similar to something like the original neural style transfer process. However, I employ a texture descriptor based on **global average pooling** across convolutional channels. This descriptor effectively captures how all convolutional filters in a (pre-trained) fully convolutional network (FCN) are activated. I first train a generative FCN (structured like an autoencoder) on a lichen photo dataset that I created. I then iteratively optimize a texture image until its texture descriptor vector matches that of a target image.
- **Result:** When learning a new texture image, my model captures the color distribution of the target image well. The spatial features in the learned image do not reflect those in the target image very well. Reasons for this are discussed.
- **In part 1/2:** I'll give a broad overview of the different paradigms in deep generative modeling, as well as some alternative texture extraction models from the literature that had better performance. This is useful since it classifies my own model and contrasts it with fundamentally different methods such as GANs.
- **In part 2/2 (this post):** I'll cover my dataset, network architecture, image learning algorithm, and results.
{: .notice--success}

# Research Objective and Brief Methods
Is it possible to take a natural image and "extract" the textures within it? I.e Can we generate images containing new instances of those textures? This **texture synthesis**, or more specifically, **texture extraction** is one of many challenging computer vision tasks that has been advanced substantially in recent years by convolutional neural networks (CNNs). The authors of [*Learning Texture Manifolds with the Periodic Spatial GAN*](https://arxiv.org/abs/1705.06566) describe it this way
> The goal of texture synthesis is to learn from a given example image a generating process, which allows to create
many images with similar properties. 

In this project I develop and evaluate a novel texture extraction method that is **image-learning-based**, meaning we optimize (learn) the pixel values of an image directly by iteratively applying gradients to the image. This procedure is in the same spirit as convolutional filter visualization and neural style transfer, covered in the sections below. 

I ask whether a novel texture descriptor based on global average pooling across all convolutional channels can capture qualitative features of the texture in a target image. 
{: .notice--warning}

The learned texture image is optimized such that, when fed into a pre-trained fully convolutional network (FCN), it activates filters in a similar way as the target image. That is, after image learning, feeding the target image and the learned image to the FCN should produce similar texture descriptor vectors. The FCN is an autoencoder that was trained to reconstruct its inputs. It was trained on a dataset of lichen macro photography that I created. In order to focus on textures, each training image is a 512 x 512 pixel patch from one of the full-sized photos.

# Dataset
<figure>
	<a href="/images/FCN_Lichen_textures/Capture_data2.png"><img src="/images/FCN_Lichen_textures/Capture_data2.png"></a>
	<figcaption>Slide borrowed from https://www.slideshare.net/andersonljason/variational-autoencoders-for-image-generation.</figcaption>
</figure>

<figure>
	<a href="/images/FCN_Lichen_textures/Capture_data1.png"><img src="/images/FCN_Lichen_textures/Capture_data1.png"></a>
	<figcaption>Slide borrowed from https://www.slideshare.net/andersonljason/variational-autoencoders-for-image-generation.</figcaption>
</figure>

<!-- ![](/images/FCN_Lichen_textures/Capture_data2.png)
![](/images/FCN_Lichen_textures/Capture_data1.png) -->

# Fully convolutional net architecture and training
![](/images/FCN_Lichen_textures/GAP.png)