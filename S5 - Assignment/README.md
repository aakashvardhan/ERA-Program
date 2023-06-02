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
- This is the input layer
```python
self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
x = F.relu(self.conv1(x), 2)
```
- Input Channel: 1 (Greyscale Image)
- Output Channel: 32
- Kernel Size: 3x3
- Activation Function: ReLU

### Convolutional Layer 2

```python
self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
x = F.relu(F.max_pool2d(self.conv2(x), 2))
```
- Input Channel: 32
- Output Channel: 64
- Kernel Size: 3x3
- Max Pool: 2x2
- Activation Function: ReLU

### Convolutional Layer 3

```python
self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
x = F.relu(self.conv3(x), 2)
```
- Input Channel: 64
- Output Channel: 128
- Kernel Size: 3x3
- Max Pool: 2x2
- Activation Function: ReLU

### Convolutional Layer 4

```python
self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
```
- Input Channel: 64
- Output Channel: 128
- Kernel Size: 3x3
- Max Pool: 2x2
- Activation Function: ReLU

### Fully Connected Layer 1

```python
self.fc1 = nn.Linear(4096, 50)
x = F.relu(self.fc1(x))
```
- Input Channel: 4096
  - This input channel is a feature map flattened(4x4x256).

- Output Channel: 50
- Activation Function: ReLU

### Fully Connected Layer 2
- This is the Output Layer
```python
self.fc2 = nn.Linear(50, 10)
x = self.fc2(x)
```
- Input Channel: 50
- Output Channel: 10
  - Number of classes(10 digits)

*The Log Softmax activation function gives the output of the 2nd fully connected layer*

## Utility Function - [utils.py](https://github.com/aakashvardhan/ERA-Program/blob/master/S5%20-%20Assignment/utils.py)

```python
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
```
- This function gives the calculation of the number of correct predictions using the predicted values and its corresponding truth label (1).


```python
def train(model,device,train_loader,optimizer,criterion):
    ...
```
- Parameters:
  - model: Neural Network class(i.e, model = Net().to(device))
  - device: Allow training the model using a specific GPU(eg: CUDA, M1.etc) or CPU
  - train_loader: "Retrieves the MNIST dataset's features and labels one sample at a time"(Cited:[Data Loader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)) (i.e, train_loader = torch.utils.data.DataLoader(train_data, **kwargs))
  - optimizer: Used to adapt the NN's attributes such as weights and learning rate, which helps reduce generalization error (i.e, optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)). 
  - criterion: Computes CrossEntropy Loss function

- This function iterates through the training MNIST data:

```python
data, target = data.to(device), target.to(device)
# The input data and its corresponding labels are moved to the GPU(i.e, mps:0)
```
```python
optimizer.zero_grad()
# Used to reset the gradient before performing backpropogation
```
    
```python
pred = model(data)
# Forward pass is done in order to obtain predictions
```

```python 
loss = criterion(pred, target)
train_loss+=loss.item()
# Calculates the Cross Entropy Loss using the predicted values and ground truth values
```
```python
loss.backward()
optimizer.step()
# Backpropogation is done by computing gradients using Stochastic Gradient Descent
```
```python
correct += GetCorrectPredCount(pred, target)
processed += len(data)

train_acc.append(100*correct/processed)
train_losses.append(train_loss/len(train_loader))

# Training Accuracy and Training loss is kept tracked
```

```python
def test(model,device,test_loader,criterion):
    ...
```
- Parameters:
  - *Explained earlier in train()*

```python
model.eval()
# Used to set to evaluation mode by not computing gradient
```
*This function has similar iteration process as train() where it obtains the predictions and calculate the test loss and test accuracy. However backpropagation is not performed.*

```python
def plt_fig(...):
  # Plots the training/testing accuracy/loss
```
