# Digit Classification using Pytorch Framework and MNIST Dataset


## Introduction

- This is an overview of the 'Net' class, a convolutional neural network(CNN) model implemented using the 'nn.module' from PyTorch's library. The architecture is designed for Image Classification.

## Architecture

### Layers 

- Convolutional Block 1 (conv1)

    - Convolution layer with 1 input channel, 8 output channels, a kernel size of 3, and padding of 1.
    - ReLU activation function.
    - Batch normalization with 8 features.
    - Dropout layer with a dropout rate of 0.1.
    - Convolution layer with 8 input channels, 4 output channels, a kernel size of 1, and padding of 1.