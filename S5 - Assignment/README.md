# Digit Classification using Pytorch Framework and MNIST Dataset

## Model Architecture - [model.py](https://github.com/aakashvardhan/ERA-Program/blob/master/S5%20-%20Assignment/model.py)

Args:
- nn.Conv2d(input_shape,hidden_units,output_shape)
input_shape: Number of Input Channels
hidden_units: Number of hidden units between layers
output_shape: Number of Output units

- nn.functional.relu(Input Tensor,inplace=False)
Applies the Rectified Linear Unit Function Element-wise 

![RELU function (Derivative)](https://www.bragitoff.com/wp-content/uploads/2021/12/mathpix-2021-12-29-18-22-34.png)

The idea of RELU is that it passes on all positive number and filters out the negative values.

- nn.functional.max_pool2d(input,kernel_size,stride=None)

Input: Input Tensor
kernel_size: Size of the Pooling Region
stride: Stride of the Pooling operation

MaxPooling is a simple solution in reducing the number of layers, which does so by increasing the receptive field.

### Convolutional Layer 1

```python
self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
x = F.relu(self.conv1(x), 2)
```
Input Channel: 1 (Greyscale Image)
Output Channel: 32
Kernel Size: 3x3
Activation Function: ReLU

### Convolutional Layer 2

```python
self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
x = F.relu(F.max_pool2d(self.conv2(x), 2))
```
Input Channel: 32
Output Channel: 64
Kernel Size: 3x3
Max Pool: 2x2
Activation Function: ReLU






