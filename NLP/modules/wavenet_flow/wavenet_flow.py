
import torch.nn as nn

import NLP.modules.conv as conv
import NLP.modules.resblock as resblock


class Wavenet_Flow(nn.Module):
    """
    Wavenet_Flow

    * Args:
        input_size:
    """

    def __init__(self, out_channels=1, num_blocks=1, num_layers=10,
                 front_channels=32, residual_channels=64, gate_channels=32, skip_channels=None,
                 kernel_size=3, cin_channels=80, causal=True):
        super(Wavenet_Flow, self).__init__()

        self.causal = causal
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.front_channels = front_channels
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cin_channels = cin_channels
        self.kernel_size = kernel_size

        self.front_conv = nn.Sequential(
            conv.ClarinetConv(1, self.residual_channels, self.front_channels, causal=self.causal),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList()
        self.res_blocks_fast = nn.ModuleList()
        for b in range(self.num_blocks):
            for n in range(self.num_layers):
                self.res_blocks.append(resblock.ResBlock(self.residual_channels, self.gate_channels, self.skip_channels,
                                                         self.kernel_size, dilation=2 ** n,
                                                         cin_channels=self.cin_channels, local_conditioning=True,
                                                         causal=self.causal, mode='SAME'))
        self.final_conv = nn.Sequential(
            nn.ReLU(),
            conv.ClarinetConv(self.skip_channels, self.skip_channels, 1, causal=self.causal),
            nn.ReLU(),
            conv.ClarinetConv(self.skip_channels, self.out_channels, 1, causal=self.causal)
        )

    def forward(self, x, c):
        return self.wavenet(x, c)
