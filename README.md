# Applied Deep Learning Projects
_________________________________

As part of my course work for my M.S. in applied mathematics at the University of Colorado, Boulder, I completed a two-semester course in applied deep learning.  The course material is publicly available and can be found [here](https://github.com/maziarraissi/Applied-Deep-Learning).  For this course, I was required to submit a "progress report" in a Jupyter notebook every two weeks.  These reports were completely open-ended.  The only requirement was that the report had to implement some of the material covered in lecture during the previous two weeks before the report was due.  Below I provide a brief description of each report.

____________________________________

**Adversarial Attacks**

This report is an implementation of the ideas presented in [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528v1).  Most adversarial attacks in computer vision look to make imperceptible perturbations of an image so that it is incorrectly classified by a neural network.  Few attack algorithms enable the attacker to target a specific class to classify an image.  In this paper, the authors introduce *adversarial saliency maps* as a way to implement such a targeted attack.  The used the LeNet architecture and MNIST dataset.  In this report, I implement the paper's algorithms on the CIFAR-10 dataset using a ResNet architecture.

____________________________________

**Autoencoders**
