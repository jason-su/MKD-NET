import torch
import torch.nn as nn
from torch.autograd import Variable

m = nn.AdaptiveAvgPool2d((1,1))
# m = nn.MaxPool1d(3, stride=2)
# m = nn.MaxPool2d(3, stride=2)
input = Variable(torch.randn(2, 4, 5))
output = m(input)

print(input.shape)
print(output.shape)

print(input)
print(output)