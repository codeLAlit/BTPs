from torch import nn
import torch.nn.functional as F
from utils import initialize_weights

class unknownClassifier(nn.Module):
    
    def __init__(self, in_channels):
        super(unknownClassifier, self).__init__()
        # output classes in, out
        self.dimr = nn.Conv2d(in_channels, 2, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        out = self.dimr(x)
        return out