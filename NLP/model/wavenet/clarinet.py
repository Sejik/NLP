
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from NLP.decorator import register
from NLP.model.base import ModelBase  # TODO
import NLP.modules.conv as conv
import NLP.modules.resblock as resblock
import NLP.modules.wavenet_flow as wavenet_flow


@register("model:teacher_clarinet")
class Teacher_ClariNet(ModelBase):
    """
    TTS Model.
    
    * Args:
    
    * Kwargs:
    """

    def __init__(
            self,
            num_blocks=2,
            num_layers=10,
            residual_channels=128,
            gate_channels=256,
            skip_channels=128,
            kernel_size=2,
            cin_channels=80,
            upsample_scales=[16, 16],
            causal=True,
            out_channels=1,
    ):
        super(Teacher_ClariNet, self).__init__()
        # TODO: model define (maybe separate the teacher and student) -> how to control the input
        # modeling

        self.causal = causal
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cin_channels = cin_channels
        self.kernel_size = kernel_size

        self.front_channels = 32
        self.front_conv = nn.Sequential(
            conv.ClarinetConv(1, self.residual_channels, self.front_channels, causal=self.causal),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        for b in range(self.num_blocks):
            for n in range(self.num_layers):
                self.res_blocks.append(resblock.ResBlock(self.residual_channels, self.gate_channels, self.skip_channels,
                                                         self.kernel_size, dilation=self.kernel_size ** n,
                                                         cin_channels=self.cin_channels, local_conditioning=True,
                                                         causal=self.causal, mode='SAME'))

        self.final_conv = nn.Sequential(
            nn.ReLU(),
            conv.ClarinetConv(self.skip_channels, self.skip_channels, 1, causal=self.causal),
            nn.ReLU(),
            conv.ClarinetConv(self.skip_channels, self.out_channels, 1, causal=self.causal)
        )

        self.upsample_conv = nn.ModuleList()
        for s in upsample_scales:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

    @overrides
    def forward(self, features, labels=None):
        # TODO: model forward
        print('')
        # c = self.upsample(c)
        # out = self.wavenet(x, c)
        # return out

        # preprocessing


@register("model:student_clarinet")
class Student_ClariNet(ModelBase):
    """
    TTS Model.

    * Args:

    * Kwargs:
    """

    def __init__(
            self,
            num_blocks=2,
            num_layers=10,
            residual_channels=128,
            gate_channels=256,
            skip_channels=128,
            kernel_size=2,
            cin_channels=80,
            causal=True,
            num_blocks_student=[1, 1, 1, 1, 1, 1],
            front_channels=32,
    ):
        super(Student_ClariNet, self).__init__()
        # TODO: model define (maybe separate the teacher and student)
        # modeling

        self.num_blocks = num_blocks_student
        self.num_flow = len(self.num_blocks)
        self.num_layers = num_layers

        self.iafs = nn.ModuleList()
        for i in range(self.num_flow):
            self.iafs.append(wavenet_flow.Wavenet_Flow(out_channels=2,
                                                       num_blocks=self.num_blocks[i], num_layers=self.num_layers,
                                                       front_channels=front_channels,
                                                       residual_channels=residual_channels,
                                                       gate_channels=gate_channels, skip_channels=skip_channels,
                                                       kernel_size=kernel_size, cin_channels=cin_channels,
                                                       causal=causal))

    @overrides
    def forward(self, features, labels=None):
        # TODO: model forward
        print('')
        # return self.iaf(z, c)

    def iaf(self, z, c_up):
        mu_tot, logs_tot = 0., 0.
        for i, iaf in enumerate(self.iafs):
            mu_logs = iaf(z, c_up)
            mu = mu_logs[:, 0:1, :-1]
            logs = mu_logs[:, 1:, :-1]
            mu_tot = mu_tot * torch.exp(logs) + mu
            logs_tot = logs_tot + logs
            z = z[:, :, 1:] * torch.exp(logs) + mu
            z = F.pad(z, pad=(1, 0), mode='constant', value=0)
        return z, mu_tot, logs_tot
