## Implementation of the article ["Fixup Initialization: Residual Learning Without Normalization"](https://arxiv.org/abs/1901.09321)


Most implementions of [ResNet](https://arxiv.org/abs/1512.03385) solve the exploding and vanishing gradient problem using Batch Normalization. However, fixup initializtion is able to solve this problem and even improve the convergence of the algorithm. 

### Experiments with MNIST dataset

The plots below illustrate the training process of ResNet50 with Batch Normalization (left) and Fixup Initialization (right). Despite the training with Batch Normalizaion is more stable, training with Fixup Initialization coverages faster and yields better accuracy. 

<img src=https://github.com/katyamineeva/fixup-initialization/blob/master/images/MNIST_resnet50_bn_loss.png alt="drawing" width="400"/> <img src=https://github.com/katyamineeva/fixup-initialization/blob/master/images/MNIST_resnet50_fixup_loss.png alt="drawing" width="400"/>

### Experiments with CIFAR-10 dataset

Experiments with CIFAR-10 confirmed the results obtained with MNIST dataset: Batch Normalization stabilizes training process, but Fixup Initialization provides better loss and accuracy. 

<img src=https://github.com/katyamineeva/fixup-initialization/blob/master/images/CIFAR10_resnet50_loss.png alt="drawing" width="400"/> <img src=https://github.com/katyamineeva/fixup-initialization/blob/master/images/CIFAR10_resnet50_acc.png alt="drawing" width="400"/>

