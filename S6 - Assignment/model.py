import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1),
            nn.Conv2d(8,4,kernel_size=1,padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(4,8,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1),
            nn.MaxPool2d(2,2)
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(8,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        )

        self.maxpool = nn.MaxPool2d(2,2)
        self.conv1d = nn.Conv2d(32,16,kernel_size=1,padding=1)

        self.conv5 = nn.Sequential(
            nn.Conv2d(16,64,kernel_size=3,padding=1),
            nn.ReLU()
        )


        self.outconv = nn.Conv2d(64,10,kernel_size=1,padding=1)
        self.gap = nn.AvgPool2d(7)





    def forward(self, x):
        x = x.to("mps:0")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv1d(self.maxpool(x))
        x = self.conv5(x)
        x = self.outconv(x)
        x = self.gap(x)
        x = x.view(-1,10)
        return F.log_softmax(x)





# old model
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #input -? OUtput? RF
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.conv5 = nn.Conv2d(256, 512, 3)
#         self.conv6 = nn.Conv2d(512, 1024, 3)
#         self.conv7 = nn.Conv2d(1024, 10, 3)

#     def forward(self, x):
#         x = x.to("mps:0")
#         x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
#         x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
#         x = F.relu(self.conv6(F.relu(self.conv5(x))))
#         x = F.relu(self.conv7(x))
#         x = x.view(-1, 10)
#         return F.log_softmax(x)