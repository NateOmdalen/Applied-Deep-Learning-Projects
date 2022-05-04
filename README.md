# Applied Deep Learning Projects
_________________________________

As part of my course work for my M.S. in applied mathematics at the University of Colorado, Boulder, I completed a two-semester course in applied deep learning.  The course material is publicly available and can be found [here](https://github.com/maziarraissi/Applied-Deep-Learning).  For this course, I was required to submit a "progress report" in a Jupyter notebook every two weeks.  These reports were completely open-ended.  The only requirement was that the report had to implement some of the material covered in lecture during the previous two weeks before the report was due.  Below I provide a brief description of each report.

____________________________________

**Adversarial Attacks**

This report is an implementation of the ideas presented in [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528v1).  Most adversarial attacks in computer vision look to make imperceptible perturbations of an image so that it is incorrectly classified by a neural network.  Few attack algorithms enable the attacker to target a specific class to classify an image.  In this paper, the authors introduce *adversarial saliency maps* as a way to implement such a targeted attack.  The paper used the LeNet architecture and MNIST dataset.  In this report, I implement the paper's algorithms on the CIFAR-10 dataset using a ResNet architecture.

____________________________________

**Autoencoders**

A generic autoencoder consists of three major components:  the encoder, the code, and the decoder.  The encoder takes the input data and produces the code.  The purpose of the code is to capture the fundamental features of the input.  The decoder takes as input the code and outputs a reconstruction of the input of the encoder.  Training an autoencoder amounts to minimizing the difference between the original input and the reconstructed output.  Therefore, the quality of the code is measured with respect to the ability of the decoder to reconstruct the original input.  In this report, I attempt to provide a geometric analysis of the quality of codes produced by autoencoders so as to obtain an alternative metric for judging the goodness of the codes produced by autoencoders.  I use the features produced by PCA as a baseline for my analysis.  The dataset used is the FashionMNIST dataset.

_____________________________________

**Convolutional Block Attention Modules**

This report provides an implementation of the ideas presented in [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521).  This paper proposes the use of CBAMs as a way to introduce an attention mechanism in convolutional neural networks.  A CBAM contains two components, a spatial and channel component.  The spatial module helps the network focus on the most relevent region of the image, while the channel attention module indentifies the most important features of the image.  In this report, I explore the effect of adding CBAMs into a ResNet-type architecture.  The dataset I use is CIFAR-10.  In addition, I use Grad-CAM to verify whether the CBAMs are in fact aiding the neural network in paying attention to the most relevant aspect of an image.

______________________________________

**CycleGAN**

This report provides an implementation of the network introduced in [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).  Image-to-image translation is the task of transforming an image from one domain to an image of another.  Typically, generative adversarial networks (GANs) are used to accomplish this task.  However, generic GANs usually require a paired dataset (e.g., an outline of a shoe and the shoe itself) for training.  CycleGAN circumvents this difficulty by modifying the typical GAN loss function.  In particular, it adds a cycle consistency constraint (going from domain A to domain B and back to domain A should produce the original image from domain A) and an identity constraint (the network should learn the identity function when given images from the target domain).  In this report, I investigate CycleGAN using the horse/zebra dataset.

_________________________________________

**Few-Shot Learning**

This report implements the ideas proposed in [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)  Few-shot learning is the task in which a classifier must learn new classes using just a few examples from each new class. The general *n*-shot *k*-way task requires a classifier to classify *n* query examples for each *k* unseen classes using a support set of examples from each class.  Prototypical networks introduce a simple inductive bias into previously used few-shot learners in order to boost their performance.  In particular, these networks work from the assumption that examples from a class cluster around a prototypical example.  The prototypical network computes this prototype in order to aid its inference.  This report investigates the performance of prototypical networks using the Omniglot dataset.

__________________________________________

**Interpretable DL**


