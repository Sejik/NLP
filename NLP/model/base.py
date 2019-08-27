
import torch.nn as nn

class ModelBase(nn.Module):
    """
    Model Base Class

    Args:

    """

    def __init__(self):
        super(ModelBase,self).__init__()

    def forward(self, inputs):
        raise NotImplementedError
