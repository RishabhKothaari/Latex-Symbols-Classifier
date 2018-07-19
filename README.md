# Latex-Symbols-Classifier
Classification of Latex symbols using Convolution Neural Networks

### About
Dataset - The HASY dataset is derived from HWRT dataset and consists of handwritten symbols. The dataset consists of 168233 instances of black and white images of handwritten symbols. Each image is of size 32x32 labeled with labels. There are 369 labels in total. The dataset can be downloaded from [HASY](https://github.com/MartinThoma/HASY).This repository makes use of supervised learning to classify latex symbols from [HASY dataset](https://github.com/MartinThoma/HASY)

### Network Structure
```
model -  Network(
  (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1))
  (conv1_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
  (conv2_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv3_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=576, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=369, bias=True)
  (tanh): Tanh()
  (prelu): PReLU(num_parameters=1)
)

```

### Accuracy

![Accuracy](https://github.com/rishab-pdx/Latex-Symbols-Classifier/blob/master/accuracy.png)

### Implementation
Programming Languages: Python3
Deep Learning Framework: PyTorch

### References
[The HASYv2 dataset
](https://arxiv.org/abs/1701.08380v1)
