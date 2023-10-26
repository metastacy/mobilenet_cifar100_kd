# Knowledge Distillation from Resnet50 to MobileNet for CIFAR100

The data used is CIFAR100, with the Decoupled Knowledge Distillation (DKD) algorithm.

The repository contains 3 .py files:
- Normal MobileNet training (achieves 0.66% accuracy).
- Self-knowledge distillation from the pre-trained ResNet50 (tba).
- Knowledge distillation from ResNet50 to MobileNet (achieves 0.71% accuracy).

The DKD code was obtained from the mdistiller repo - https://github.com/megvii-research/mdistiller
The pre-trained ResNet50 was obtained from - https://huggingface.co/edadaltocg/resnet50_cifar100
