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

This report compares VGG11 and VGG16 via temperature scaling and layer-wise relevance propagation (LRP).  Temperature scaling is a calibration technique introduced in [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).  Many modern neural networks are poorly calibrated, meaning that the distribution over classes that it learns is highly biased.  Temperature scaling attempts to remedy this problem by dividing the logits by a "temperature" parameter.  LRP is presented in [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10) and is a method that identifies the most relevant input features for classification by propagating the prediction back into the network according to a certain set of rules.

__________________________________________

**Language Modeling**

BERT is a very popular langauge model that was introduced in [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).  In this report, I attempt to train from scratch a small BERT model and a large BERT model using the wikitext dataset.  Language models are only as good as the downstream tasks that they support.  Therefore, as a way to compare the models, I use transfer learning to train each model to perform sentiment analysis on the emotions dataset (found [here](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)).  I use a "fully" pre-trained BERT model as a baseline to compare the performance of the two models I pre-train.

___________________________________________

**Mixup**

This report investigates the claims made in [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412).  Supervised learning can be characterized as the task of minimizing the expected risk of misclassifying an example.  This requires the specification of a loss function and an underlying distribution for the data. Typically, since little is known in advance about the data, one implicitly assumes a Dirac distribution.  The resulting model is called an empirical risk minimization model.  This paper argues that the mixup distribution provides a better model of the data.  This distribution is an example of a "vicinal distribution."  Sampling is achieved by taking convex combinations of images from different domains.  In this report, I investigate whether this distribution is a good model for the geometry of the space of images.  I use the CIFAR-10 dataset.  I also compare this method to another vicinal distribution that assumes the data is Gaussian.

___________________________________________

**Neural Collaborative Filtering**

This report explores the recommender system proposed in [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031).  Collaborative filtering models a user's preferences via their interactions with the items that may be potentially recommended.  Neural Collaborative Filtering (NCF) models these interactions with a neural network.  The paper demonstrates that traditional filtering methods such as matrix factorization can be generalized to an NCF model.  The paper terms these models generalized marix factorization (GMF) models.  It also proposes a NeuMF model, which combines a GMF model with an MLP model.  The resulting network is unique in that it has both a shallow component and a deep component, which helps the network from overfitting.  In this report, I investigate NCF models using the MovieLens dataset.

___________________________________________

**Neural Machine Translation**

This report explores seq2seq models introduced in [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215).  Previous efforts in machine translation needed both deep learning and classical machine learning in order to map sequences to sequences.  This paper was one of the first breakthroughs in developing neural networks that perform the entire translation without any aid from classical ML.  In this report, I train a number of seq2seq models using the Multi30k dataset and evaluate them with the BLEU score.  In particular, I investigate the performance of different kinds of architectures, both small and large.  I also investigate the effects of the smoothing method used in the BLEU score on our ability to interpret the test results.
___________________________________________

**Object Detection**

This report implements the Faster R-CNN object detection model presented in [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497).  The the innovative idea of this paper is the design a fully-convolutional region proposal network that shares parameters with the detection network thereby making the network faster than the Fast R-CNN.  This report implements this model on the LaRA traffic light dataset.  Of particular interest in this report is the tuning of the intersection-over-union threshold.  This is accomplished by looking at the precision-recall curves and choosing the threshold that best balances precision and recall.
____________________________________________

**Sentiment Analysis with CNNs**

Prior to the development of large language models, researchers experimented with using convolutional neural networks for text classification.  This report explores one such network introduced in [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).  Using the IMDb dataset, I train this network to classify movie reviews as either positive or negative.  The weights for the word embedding were pre-trained using word2vec.  I also investigate the effect of the dropout probability on the performance and compare networks that are trained with and without static learning.  In static learning, the word embedding is frozen during training, while in non-static learning, the weights of the word embedding are free to be updated. 
_____________________________________________
