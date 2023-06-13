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

- Convolutional Block 2 (conv2)

    - Convolution layer with 4 input channels, 8 output channels, a kernel size of 3, and padding of 1.
    - ReLU activation function.
    - Batch normalization with 8 features.
    - Dropout layer with a dropout rate of 0.1.
    - Max pooling layer with a kernel size of 2 and a stride of 2.

- Convolutional Block 3 (conv3)

    - Convolution layer with 8 input channels, 16 output channels, a kernel size of 3, and padding of 1.
    - ReLU activation function.
    - Batch normalization with 16 features.
    - Dropout layer with a dropout rate of 0.1.

- Convolutional Block 4 (conv4)

    - Convolution layer with 16 input channels, 32 output channels, a kernel size of 3, and padding of 1.
    - ReLU activation function.
    - Batch normalization with 32 features.
    - Dropout layer with a dropout rate of 0.1.

- Max Pooling (maxpool)

    - Max pooling layer with a kernel size of 2 and a stride of 2.

- 1x1 Convolution (conv1d)

    - Convolution layer with 32 input channels, 16 output channels, a kernel size of 1, and padding of 1.

- Convolutional Block 5 (conv5)

    - Convolution layer with 16 input channels, 64 output channels, a kernel size of 3, and padding of 1.
    - ReLU activation function.

- Output Convolution (outconv)

    - Convolution layer with 64 input channels, 10 output channels, a kernel size of 1, and padding of 1.

- Global Average Pooling (gap)

    - Average pooling layer with a kernel size of 7.


### Forward Pass

The forward pass function moves the input tensor through the layers as followed:

1. The tensor is moved to the "mps:0" device.
2. The tensor is passed through conv1.
3. The tensor is passed through conv2.
4. The tensor is passed through conv3.
5. The tensor is passed through conv4.
6. The tensor is passed through the maxpool layer, and the result is passed through conv1d.
7. The tensor is passed through conv5.
8. The tensor is passed through outconv.
9. The tensor is passed through gap.
10. The tensor is reshaped using view(-1,10).
11. The log_softmax function is applied to the tensor to get the output probabilities.