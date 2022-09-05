import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_size, in_features, out_size):
        super().__init__()
        self.conv1 = nn.Conv1d(
          in_features,
          50,
          kernel_size=21,
          stride=1,
          padding=10
        )
        self.conv2 = nn.Conv1d(
          self.conv1.out_channels,
          25,
          kernel_size=11,
          stride=1,
          padding=5
        )
        self.conv3 = nn.Conv1d(
          self.conv2.out_channels,
          20,
          kernel_size=7,
          stride=1,
          padding=3
        )
        self.conv4 = nn.Conv1d(
          self.conv3.out_channels,
          1,
          kernel_size=1,
          stride=1,
          padding=0
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        x = x.flatten(start_dim=1)
        return x