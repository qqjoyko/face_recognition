from torch import nn
import torch.nn.functional

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # The first layer of convolution->activation->pooling->Dropout
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(  # The second layer of convolution->activation->pooling->Dropout
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.out = nn.Linear(32 * 8 * 8, 2)  # Fully connected layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return nn.functional.log_softmax(x, dim=1)  # Log + softmax the result and output
