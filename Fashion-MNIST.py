import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # TODO
        # implement the forward pass
        return t


network = Network()
print(network)
# The printed comes from override in Pytorch __repr__ method
# Network(
#     (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
#     (conv2): Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1))
#     (fc1): Linear(in_features=192, out_features=120, bias=True)
#     (fc2): Linear(in_features=120, out_features=60, bias=True)
#     (out): Linear(in_features=60, out_features=10, bias=True)
# )
# Notice the kernel and stride defaults to a square
# Notice bias can be turned off
