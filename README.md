# Applied Deep Learning Projects
_________________________________

As part of my course work for my M.S. in applied mathematics at the University of Colorado, Boulder, I completed a two-semester course in applied deep learning.  The course material is publicly available and can be found [here](https://github.com/maziarraissi/Applied-Deep-Learning).  For this course, I was required to submit a "progress report" in a Jupyter notebook every two weeks.  These reports were completely open-ended.  The only requirement was that the report had to implement some of the material covered in lecture during the previous two weeks before the report was due.  Below I provide a brief description of each report.

____________________________________

**Adversarial Attacks**

This report is an implementation of the ideas presented in [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528v1).  Most adversarial attacks in computer vision look to make imperceptible perturbations of an image so that it is incorrectly classified by a neural network.  Few attack algorithms enable the attacker to target a specific class to classify an image.  In this paper, the authors introduce *adversarial saliency maps* as a way to implement such a targeted attack.  The used the LeNet architecture and MNIST dataset.  In this report, I implement the paper's algorithms on the CIFAR-10 dataset using a ResNet architecture.

____________________________________

**Autoencoders**

A generic autoencoder consists of three major components:  the encoder, the code, and the decoder.  The encoder takes the input data and produces the code.  The purpose of the code is to capture the fundamental features of the input.  The decoder takes as input the code and outputs a reconstruction of the input of the encoder.  Training an autoencoder amounts to minimizing the difference between the original input and the reconstructed output.  Therefore, the quality of the code is measured with respect to the ability of the decoder to reconstruct the original input.  In this report, I attempt to provide a geometric analysis of the quality of codes produced by autoencoders so as to obtain an alternative metric for judging the goodness of the codes produced by autoencoders.  I use the features produced by PCA as a baseline for my analysis.  The dataset used is the FashionMNIST dataset.

_____________________________________

**Convolutional Block Attention Modules**
