import torch
import torch.nn as nn

fc = nn.Linear(in_features=4, out_features=3) #Set Breakpoint here

t = torch.tensor([1,2,3,4], dtype=torch.float32)
#instead of calling the forward method direction, you instanciate the Linear function and it will create and update the weights for you, and automatically add the forward method in the call
output = fc(t)

print(output)