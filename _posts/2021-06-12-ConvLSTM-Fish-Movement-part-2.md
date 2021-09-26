--- 
title: "Predicting Fish Movement with Convolutional LSTMs part 2/5"
date: 2021-06-12
# layout: "single"
permalink: /ConvLSTM_Fish_2/
categories:
    - Computer Vision
    - Deep Learning
    - Animal Behavior
excerpt: "Outline of Project Subtasks"
tagline: "Outline of Project Subtasks"
header:
  overlay_image: /images/ConvLSTM_forcasting/banner_crop_dark.png
mathjax: "true"
toc: true
toc_label: "Contents"
---
See [part 1](/ConvLSTM_Fish_1/) for an overview of this project and links to the github repository.
{: .notice--warning}

This post is a list and brief overview of all subtasks that I needed to do to complete this project, in roughly the order they were done. If you are new to deep learning, many of these may not be obvious! They weren't to me. As Andrew Ng has said (paraphrased), 
>"[most of the work in developing AI is dirty work (e.g. data engineering/cleaning, debugging code)]". 

In the next posts I'll go into more depth for many of these, including the math notation where needed.

# Data Taking
The first step is of course obtaining video data. I placed a DSLR camera overhead of my fish observation tank and recorded at 30fps for 30 mins while I was not in the room, at constant ambient temperature and light. It's important at each stage to get as good, clean data as possible. I chose an observation tank with completely white background and removed any plants before filming that could interfere with the model learning fish body shapes.

# Preparing the dataset
* I used Sony Movie Studio Platinum 16 to import the video and export a set of ~54000 frames. I also used the program to perform image processing steps such as improving contrast and making the background a more uniform shade.
* I divided the images into a training (first 83.38% of images), validation (following 6.66% of images), and testing (last 9.95% of images) dataset. Each set needs to be placed into a separate folder.

# Obtain ground truth labels: Converting into a Self-supervised setting
We are doing frame-by-frame binary classification and therefore need a way to label each frame. To do this I converted the problem into a **self-supervised** setting. Self-supervised strategies generally involve either 1. leveraging domain knowledge to derive suitable labels for unlabeled data, or 2. covering up some part of the data and asking the model to recover the missing part. In the latter case the label or target may be the original unaltered data itself. Examples of this type include image colorization, image in-painting, text completion training strategies in NLP, or the approach I took here. 

>I effectively use information about the future dynamics as a label for the past. 

I use a proxy for the group mean swimming speed called $$\tilde{y_i}$$. $$\tilde{y_i}$$ basically measures the absolute change in pixels between frames $$i$$ and $$i+1$$. The faster the fish move, the more pixels have changed between consecutive frames. I look at the dynamics of $$\tilde{y_i}$$ occurring *after* frame $$i$$ to label the video clip *ending* at frame $$i$$. The labeling algorithm can be found in [part 3](/ConvLSTM_Fish_3/). Roughly, I label a frame $$i$$ as 1 if a) $$\tilde{y_i}$$ is relatively low and b) it preceded a sudden rise in $$\tilde{y_i}$$, and label as 0 if it did not meet these conditions. $$\tilde{y_i}$$ will be defined formally in the next post.

# Make a baseline model as a benchmark
It's important to be able to compare your trained model with a simpler or more naïve approach. This baseline model can be a simpler machine learning model, or in my case, an algorithm producing probabilities or classifications based on heuristics.

I created two baseline models that each use only the timeseries $$\tilde{y_i}$$ to make frame classifications. The first is purely algorithmic and outputs the probability that frame $$i$$ has label $$y_i = 1$$. This algorithm, `baseline_prediction_probabilistic`, can be found in the notebook `Compute_targets-time-channels.ipynb`. Broadly, it takes a window of about 90 frames preceding frame $$i$$ and produces high probability if $$\tilde{y_i}$$ is both low at the end of this window and the range of $$\tilde{y_i}$$ is high across the window (which suggests recent aggressive activity).

The second baseline model is a standard 2-layer LSTM which is trained on the $$\tilde{y_i}$$ timeseries. Its dataset consists in windows of equal length (4 sec) as the video clips given to the ConvLSTM. The python file building and training this model is called `train_y_tilde_LSTM.py`.
# Write data input pipeline/Data generators
*This step is one of the most important and can not be underestimated!* I'll use my model to illustrate. The ConvLSTM takes an input of shape `(time-steps x width x height x channels)`. This is obviously more complicated than the input given to simple CNNs `(width x height x channels)`. To my knowledge, Keras does not have pre-built image data generators (for e.g. `tf.keras.preprocessing.image.ImageDataGenerator` + `data_generator_name.flow_from_directory` or `tf.keras.preprocessing.image_dataset_from_directory`) that can produce ConvLSTM inputs. 

I decided to write my own **python generator** that would assemble images from the training or validation directory into tensors of the correct shape. It is called `tc_inputs_generator` in `train_ConvLSTM_time_channels_with_augmentation.py`. I then used `tf.data.Dataset.from_generator(tc_inputs_generator,...)` To create the `tf.data.Dataset` object that is actually passed to `model.fit()`. 


I had tried several methods that failed before figuring this out. But this process demonstrates the flexibility of `tf.data.Dataset` data generators to create any kind of input pipeline you want. I also used `tf.data.Dataset`s to shuffle, batch, and even augment the dataset.

Finally, it's important to note that the input data must be generated on-the-fly like this, because it would waste a tremendous amount of memory if we had to store each input tensor separately. Since **there is one input tensor per frame** in the dataset, there would be over 90% overlap in data between the tensors of any consecutive frames. 
# Build your model
This is the obvious step and it's what I imagine most people think about when the phrase "doing deep learning" comes up. Here you design the architecture of the model. In tensorflow/Keras functional API it can be called "building the computational graph." This graph is a directed acyclic graph where each "node" is a "layer." There can be no loops (acyclic), but there can be multiple inputs, multiple outputs, merges between branches, and other exotic topological features that I'm probably forgetting. A keras "layer" is really a combination of multiple things in one. It is 
1. a mathematical operation mapping some input to an output. The shapes and dimensionality of the input need not equal the output.
2. a set of trainable (i.e. can be updated during optimization) and untrainable parameters. These parameters define or parametrize the mathematical function
3. an associated output tensor

The layer object contains other attributes such as `name` and `input_shape` that you can look up in the documentation.

# Hyperparameter tuning
The time this can take and heartache it can cause can also **not** be underestimated. The ability to get good accuracy on the test set turned out to be shockingly sensitive to many factors such as the learning rate, batch size, model size, class weights, regularization, and data augmentation.
{: .notice--warning}

Here is where you adjust things like learning rate and the size and number of layers in the model. It is best practice to start with a very small model, check that it can at least overfit on the training set (or a fraction of the training set), and progressively scale up the model. If you are not doing transfer learning (i.e. building a model from scratch), it is tempting to start with a large fancy model. Don't. I've found that the 'capacity' of CNNs to learn complex mappings is often way more than you would think. For example, I believe VGGnet or Alexnet was trained to high accuracy on Imagenet even **when all class labels were randomly swapped!** That is fascinating because it illustrates the capacity of these models to learn extremely turbulent mappings.

In the future, autoML methods such as neural architecture search and hyperparameter optimization will eventually make this step obsolete. If that one day becomes the norm, the field of AI will have once again elevated into a new paradigm, where researchers spend their time on different sorts of problems than what they do now. The more recent paradigm shift came with the advent of **automatic differentiation**. This allowed the gradient of the loss function with respect to the weights within some arbitrary computational graph to happen automatically. Whereas before, backpropagation had to be computed manually. This allowed researches to effectively work at a higher level, where components of models could be modularized and assembled like Legos. And you can now simply ask the gradient to be computed. Needless to say, this accelerated progress in the field tremendously.

# Train model
This step goes hand-in-hand with hyperparameter tuning. In my project I did not use transfer learning so all weights had to be learned from scratch. Transfer learning and/or fine tuning is recommended if the dataset you are working with has some similarity to another pre-trained model's dataset. Usually the earliest layers of a CNN, which act as edge/color detectors can be imported into a new model and the higher layers' weights an be trained from scratch or fine tuned. 

It is good practice to dedicate a small portion of your dataset to being a **validation set**. The reason is that you need to be able to monitor the model's performance on this unseen data and the training data at the same time during training. If everything is going well, the loss on both sets will begin dropping and at a certain point, the model will begin to overfit on the training set. We can tell this is happening because the training loss will begin to fall below the validation loss, and the validation loss may even start to increase after that point.

# Troubleshooting - Things I tried when validation loss wasn't improving
I was monitoring the training and validation ROC AUC since I'm training a binary classifier, and quickly found that the model can rapidly overfit on the training set. The smaller the training set, the faster it can overfit. The validation AUC would remain at 50% under many of the conditions I tried. There were also many training runs where **the model learned to output the exact same value for any input**. If your dataset has high class imbalance (such as 90% of samples are labeled negative), the Accuracy metric may be misleading (If the model classifies everything as negative, it would have 90% accuracy). Its important to check multiple metrics during training. 

Adjusting all these factors can have an impact on the model's ability to generalize to the test set:
* learning rate
* batch size
* model size
* class weights or other methods to ameliorate class imbalance in the dataset
* type and amount of regularization
* data augmentation


# Write code for measuring performance and visualizing model behavior
* For example, Plot ROC curve and measure AUC
* Generate results figures
* Ablation studies

# Writing up your results! Publish a paper, blog, or make a YouTube video.
The paper you're reading now.