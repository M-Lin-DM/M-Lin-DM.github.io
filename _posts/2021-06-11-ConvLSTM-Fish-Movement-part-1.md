--- 
title: "Predicting Fish Movement with Convolutional LSTMs part 1/5"
date: 2021-06-11
# layout: "single"
permalink: /ConvLSTM_Fish_1/
categories:
    - Computer Vision
    - Deep Learning
    - Animal Behavior
excerpt: "Project motivation, guide to the github repository, and tools used"
tagline: "Introduction, Tools used, and Github Repository"
header:
  overlay_image: /images/ConvLSTM_forcasting/banner_crop_dark.png
mathjax: "true"
toc: true
toc_label: "Contents"
---

>**Can a deep neural net known as the Convolutional LSTM forecast aggressive interactions in a group of *Serpae tetra* fish? If so, it may be useful in modeling other systems with complex spatio-temporal dynamics.**

**Project In Brief - TLDR** 
- Under my laboratory conditions, *Serpae tetra* fish exhibit aperiodic, "bursty" movement patterns---typically, one fish begins chasing another when the two get too close together. [[video]](link)
- **I hypothesized that**  ConvLSTM would be particularly well-suited to model these dynamics. The underlying assumption is that the fishes' prior dynamics encodes some information on if/when a burst is likely to occur.
<!-- - To obtain ground truth labels, I converted this problem into a *self-supervised* setting. Roughly, each frame $$i$$ was algorithmically labeled as 1 if the group mean speed $$\tilde{y_i}$$ was low and frame $$i$$ preceded a sufficiently large increase in $$\tilde{y_i}$$ across frames $$[i, i+45]$$. Conversely, frame $$i$$ was labeled as 0 if it did not precede a large increase in speed. The method of computing $$\tilde{y_i}$$ and the full details of the labeling algorithm can be found in [part 3](/ConvLSTM_Fish_3/) -->
- **Research Goal: Train a ConvLSTM-based model that can take a video clip of fish as input and predict whether or not a burst of motion will occur during the subsequent 1.5 seconds.** (Visually, this burst corresponds to sudden aggressive interactions between two or more fish and a spike in group mean speed.)
- **Methods:** I first filmed a group of 6 *Serpae tetra* fish from overhead, for 30 mins. I converted this into an image dataset containing ~54000 grayscale frames with 120x160x1 resolution. I trained a ConvLSTM-based model to act as a **frame-by-frame binary classifier**. This recurrent neural network takes as input a 4 second clip of video frames (stacked as a single 4D tensor of shape (time-steps x width x height x channels)) and outputs the probability that a 'spike' in group mean velocity will occur during the next 1.5 seconds (45 frames).
- **Result:** The model does fairly well at identifying frames that come right before the fish chase each other. It achieves an ROC-AUC of 0.83 and a peak accuracy of 84.7%, matching the algorithmic, baseline classifier (AUC: 0.84 Peak accuracy: 85.4%). Using a probability threshold that maximizes Youden's statistic, the model achieved a true positive rate of 82.9%, with a false positive rate of 31.0%. 
<!-- This helps confirm that the fishes' prior spatial configurations and dynamics contain information that is useful in predicting the onset of aggressive interactions. -->
- This 5-part series is a hybrid between an academic paper and an informal guide to doing a deep learning research project. I will
1. Introduce the problem we're trying to solve and explain why deep learning/ConvLSTMs constitute a novel approach to it. (This post) 
2. Report my methods and findings in detail
3. Explain key parts of my python-keras-tensorflow code.
4. Provide a step-by-step guide outlining the subtasks required to take a project like this from start to finish.  
{: .notice--success}

# Guide to Github Respository: "Convolutional-LSTMs-for-Motion-Forcasting"
- Repository for this project: [Github](https://github.com/M-Lin-DM/Convolutional-LSTMs-for-Motion-Forcasting)
- `train_ConvLSTM_time_channels_with_augmentation.py` -- Python file for building and training the ConvLSTM and creating the inputs data generator
- `Compute_targets.ipynb` -- Jupyter notebook for generating ground truth labels (for single-channel version of the model)
- `Compute_targets-time-channels.ipynb` -- Jupyter notebook for generating ground truth labels (for "time-channels" version of the model) and Baseline (algorithmic) model
- `Analyze_performance-tc-test-data.py` -- Python file for analyzing performance on the test set
- `Analyze_performance-tc-train-data.py` -- Python file for analyzing performance on a portion of the training set
- `Evaluate_on_test_set.ipynb` -- Jupyter notebook for making results figures such as ROC and time-series plots.
{: .notice--warning}


# Introduction
## The challenges of modeling spatio-temporal dynamics | existing approaches
Forecasting systems with complex spatio-temporal dynamics, such as collective animal or insect motion, is a challenging problem for several reasons. Examples of such systems include sheep herding, fish schooling, bird flocking, territorial combat in hamadryas baboons, emergency stampedes in human crowds, and alarm signal spread in ant colonies. Some factors that make these systems difficult to study are the following:

1. Data acquisition can be challenging. Several modalities exist for tracking individuals and none is perfect. For example, GPS or RFID tag-based methods are often limited in either spatial or temporal resolution. Computer vision-based methods like object tracking from raw video or fiducial markers may have better spatial and temporal resolution, but must deal with inaccuracies arising from occlusion and variance in environmental lighting or viewing angle.

2. An individual agent's internal decision-making process is generally impossible to observe. Only its behavior, the outcome of this process, is observable.

3. The consequences of multiple agents' combined interactions are difficult to predict. One hand-waving statement, that I've heard people use to explain this, is that the cumulative effect of these interactions tends to be a *non-linear* function of all agents' actions, rather than a simple sum. I still don't have the clearest picture of what that means but I'll leave it in case you do.

Several approaches have been applied to modeling collective motion, and spatio-temporal dynamics in general. For example, *partial differential equations* can be used when the quantity of interest is something continuous like density, chemical concentration, or heat, over space. Fascinating chaotic spatio-temporal dynamics such as "turning patterns" can result from such PDE-based methods (See videos of ["reaction-diffusion" models](https://www.youtube.com/user/Softology/videos)). *Agent-based models* have also been used to control the individual behavioral rules of each agent or component in a system. In these simulations, agents are able to make decisions independently and interact with each other. A related topic is *self-propelled particle simulations*, which have been extremely successful at modeling collective motion such as [bird flocking](https://www.youtube.com/watch?v=QbUPfMXXQIY) [3D example](https://www.youtube.com/watch?v=_5tJ8jwd64Y) or fish schooling.) Finally, [*cellular automata*](https://www.youtube.com/watch?v=C2vgICfQawE) use simple rules to iteratively update local regions of space, and such simulations can lead to extremely complex spatio-temporal dynamics as well.

>In this project, I use deep neural networks to forecast aggressive interactions in a group of 6 *Serpae tetra* fish. After observing these fish, I had an intuition that a type of recurrent neural network called a **convolutional LSTM** might be particularly good at modeling their dynamics. I wanted to test the hypothesis that the fishes' motion could be predicted by this model. If this turned out to be true, the model could be a useful forecaster in other applications with complex spatio-temporal dynamics. 

## Why the Convolutional LSTM is the perfect fit for this problem

### ConvLSTMs in Video synthesis
One of the ConvLSTM's main applications that I've seen so far has been in **video synthesis** problems.  Many papers have used the "moving MNIST" dataset as a benchmark. This is a video dataset where hand-written digits move linearly across a square, grayscale image, and when they reach the edge, bounce off at an equal angle of incidence. These simple dynamics establish a good benchmark for video prediction tasks because the prior video frames clearly encode some information on how the dynamics will evolve. A recurrent neural net  should, in theory, be capable of learning to extrapolate into the future. The computer vision task here is to synthesize future frames of video given a seed sequence. 

The ConvLSTM functions much like a traditional LSTM in that it loops over each frame in the input frame sequence, and it can be trained to generate the very next frame or a sequence of future frames. Thus, ConvLSTM can be considered a **generative model** (when adapted to this task).

>For my experiment, I chose not to attempt video synthesis. I adapted the ConvLSTM to perform binary classification on a frame-by-frame basis. 

**Relevant Articles in video synthesis**
1. [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting (2015)](https://arxiv.org/abs/1506.04214)
2. [Unsupervised Learning of Video Representations using LSTMs (2015)](http://proceedings.mlr.press/v37/srivastava15.html)
3. [PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs (2017)](https://dl.acm.org/doi/abs/10.5555/3294771.3294855)

### Holistic vs Distributed modeling of spatio-temporal dynamics

One advantage of the ConvLSTM is that it allows us to model spatio-temporal dynamics more *holistically*. In contrast to several of the methods mentioned in the first section, the deep learning-based approach taken here is more holistic in the sense that it forms, at each time-step, a representation of the system's state *as a whole*, rather than modeling each of its components individually. I'll come back to this idea in a bit.

In my dissertation work, I took the latter approach. I used [object-tracking to obtain the 2D spatial trajectory of each ant(video)](https://www.youtube.com/watch?v=RumJ0G47BdM). I then used agent-based models to simulate their movement in a bounded 2D space. Under these kind of methods, we never form a *unified* representation of the ant colony. The system instead has a *distributed* representation: namely, the set of all ants' positions, velocity vectors, body orientations, and other individual-specific information. 

In many systems with complex spatio-temporal dynamics (e.g. the rippling surface of a lake, a dynamic Turing pattern, [migrating wildebeest herds](https://www.youtube.com/watch?v=u3JcudYt8GU), or the shifting distribution of homeless encampments in a city), it can be difficult to form a distributed representation like the one I described for ant colonies. At the same time, it is not obvious how one would hand craft a feature vector that usefully represents the state of the system as a whole. One naive approach would be to flatten each video frame (say of size N x N) into a feature vector with $$N^2$$ elements. More often than not, the extremely high dimensionality of such a vector would limit its use in unsupervised methods like clustering---let alone forecasting. 
>The ConvLSTM, and deep learning in general, offer a compelling solution: **learn** a unified representation of the system by exposing a deep neural net to a large number of system states.

### ConvLSTMs combine spatial and temporal representation learning

#### What is representation learning?
>Representation learning is a set of techniques that allows a system to automatically discover the representations needed for feature detection or classification from raw data. This replaces manual feature engineering and allows a machine to both learn the features and use them to perform a specific task. -[Wikipedia](https://en.wikipedia.org/wiki/Feature_learning)

In my own words, "representation learning" is really what defines the departure from traditional, hand-crafted feature engineering. The beauty and efficiency of deep learning is that (with perceptual problems) we no longer have to figure out how to represent the data as feature vectors. We can now feed a model with just raw data such as images, audio, or text, and through the process of training, it will learn how to represent the features in that data most effectively. That is representation learning.

A key idea explored in this project is that, by training on a large corpus of raw data, ***the ConvLSTM can learn powerful feature representations that simultaneously capture both spatial structures and temporal dynamics.*** 

For analogy, convolutional nets (CNNs) achieve representation learning by automatically learning convolutional kernels that act as detectors for objects commonly occurring in the dataset. For example, in early CNN layers you'll find kernels that have been optimized to detect low-level features such as edges or blobs. At higher convolutional layers, kernels will detect more complex structures such as parts of faces or cars (that is, *if* faces and cars were present in the dataset and the model was trained for, say, classification). 

#### ConvLSTMs are recurrent networks that can learn to represent spatial structures
ConvLSTM augments CNN's spatial representation learning by including a recurrent mechanism. Or conversely, it is a recurrent neural net that contains 2D convolution kernels. It accumulates system state information within its hidden state and memory cell while moving across a time-series. Like an LSTM, this accumulation is mediated through its gates. The ConvLSTM **gates** operate in a identical way as in LSTM: they determine what information is stored/passed on to the next step, forgotten, and output at each step.

By effectively hybridizing convolutional networks with recurrent networks the ConvLSTM draws upon the strengths of each. It is essentially an LSTM that uses convolutions in place of the usual densely connected layers. At each recurrent step, the input, hidden state, and memory cell are all 3D tensors instead of feature vectors (as in a regular LSTM). 

>Just as a regular LSTM loops down a stack of vectors (computing a hidden state and updating a memory cell at each step), the ConvLSTM loops down a stack of **images**.

It likewise computes a hidden state (via *convolutions*) and updates a memory cell at each step.

As a layer in a model, the ConvLSTM cell can either output all hidden states along the loop, or just the very last state, depending on one's architectural choice. 

This all has been a very high-level description and a more detailed view of the model will be given in [part 4](/ConvLSTM_Fish_4/). If it doesn't make sense, I wouldn't think too hard about it just yet!

From Shi *et al.* 2015, *"Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"*:
> The major drawback of FC-LSTM in handling spatiotemporal data is its usage of [fully connected layers] in input-to-state and state-to-state transitions in which no spatial information is encoded... Our input and output elements are all 3D tensors which preserve all the spatial information. Since the network has multiple stacked ConvLSTM layers, it has strong representational power which makes it suitable for giving predictions in complex dynamical systems like the precipitation nowcasting problem we study here. 


### Central hypothesis and assumption
Coming back to our initial goal, we want to provide a deep neural network with short video clip of fish motion, and we want it to predict whether a burst of motion (defined in terms of a speed spike) is going to occur within the next 1.5 sec. The model should ultimately be able to provide a signal, ahead of time, that an aggressive interaction is about to take place.
My intuition was that, if the ConvLSTM can form feature representations that combine spatial and temporal dynamics information, it could learn to identify signatures/motifs in the fishes' spatial configurations and dynamics that tend to precede bursts of motion. My underlying assumption is that the fishes' prior dynamics encodes some information on if/when a burst is likely to occur.


### Summary
In summary, the ConvLSTM is an advantageous approach in cases where you have complex spatio-temporal dynamics, and it is difficult to represent the state of the system as a feature vector. This is true for natural images and is certainly true for most videos. The ConvLSTM is also potentially useful for modeling spatio-temporal dynamics because it is capable of combined spatial and temporal representation learning. The feature representations it learns can thus be used to describe the state of a video-recorded system in a holistic way; i.e. representing the system's state, as a whole, as one feature vector or tensor. I believe this represents a novel modeling approach, when it comes to forecasting complex spatio-temporal dynamics. 

# Up Next
In [part 2](/ConvLSTM_Fish_2/) I'll outline the main subtasks needed to take this project from start to finish. For example, data engineering, building and training the model, creating benchmarks, and performing analysis.

# Hardware, software, and programs used
**Hardware**
- OS: Microsoft Windows 10 Home
- Graphics card: one NVIDIA RTX 2080Ti, 11GB
- CPU: AMD Ryzen 9 3900X 12-Core Processor 3.80 GHz
- Camera: Cannon EOS 80D DSLR

**Programming, writing, publishing**
- Python 3.7
- Pycharm
- Jupyter Notebook
- Visual Studio Code
- Github Pages
- Latex, TexStudio
- Anaconda Navigator

**Deep Learning**
- Tensorflow 2.3.0
- Keras 2.4.3
- CUDnn 7.6.5
- [Weights and Biases](https://wandb.ai/site) (experiment tracking)

**Plotting and Data Visualization**
- Plotly v (data visualization)
- [Sony Movie Studio Platium 16](https://www.vegascreativesoftware.com/index.php?id=375&L=52&AffiliateID=147&phash=9AEBDNMCLOZchLqc&utm_source=Bing&utm_medium=cpc&utm_campaign=Brand_Vegas_Movie_Studio_Platinum_US&ef_id=206c1a909a7e1971bb6c7a533e8280be:G:s&msclkid=206c1a909a7e1971bb6c7a533e8280be)
- [Fiji (ImageJ)](https://fiji.sc/)
- [Sketchup](https://www.sketchup.com/)
- Microsoft Powerpoint
