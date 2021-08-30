import torch.nn as nn
from downsample import Downsample
from torch.nn import Dropout
from torch.nn.utils import spectral_norm
from torch.nn import ModuleList
import torch.nn.functional as F
import torch

def swish(x):
    return x * torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, args, downsample=True, rescale=True, filters=64, \
        norm=True, spec_norm=False, alias=False):
        super(ResBlock, self).__init__()

        self.filters = filters
        self.downsample = downsample

        if filters <= 128:
            self.bn1 = nn.InstanceNorm2d(filters, affine=True)
        else:
            self.bn1 = nn.GroupNorm(32, filters)

        if not norm:
            self.bn1 = None

        self.args = args

        if spec_norm:
            self.conv1 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
        else:
            # self.conv1 = WSConv2d(filters, filters, kernel_size=3, stride=1, padding=1)
            self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        if filters <= 128:
            self.bn2 = nn.InstanceNorm2d(filters, affine=True)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=True)

        if not norm:
            self.bn2 = None

        if spec_norm:
            self.conv2 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
        else:
            # self.conv2 = WSConv2d(filters, filters, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(0.2)

        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ELU(inplace=True)
        self.act = swish

        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)

                if alias:
                    self.avg_pool = Downsample(channels=2*filters)
                else:
                    self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

                if alias:
                    self.avg_pool = Downsample(channels=filters)
                else:
                    self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)


    def forward(self, x):
        x_orig = x
        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)

        if self.bn2 is not None:
            x = self.bn2(x)

        # will this be necessary?
        x = x + x_orig
        x = self.act(x)

        x_out = x 

        if self.downsample:
            x_out = self.conv_downsample(x_out)
            x_out = self.act(self.avg_pool(x_out))

        return x_out


class ResNetModel(nn.Module):
    def __init__(self, args, stride=2):
        super(ResNetModel, self).__init__()
        self.act = swish

        self.args = args
        self.spec_norm = False
        self.norm = True

        # self.relu = torch.nn.ReLU(inplace=True)
        self.relu = nn.ELU(inplace=True)
        self.downsample = Downsample(channels=3)

        filter_dim = 64

        self.conv1 = nn.Conv2d(3, filter_dim, kernel_size=7, stride=stride, padding=3)

        self.res_1a = ResBlock(args, filters=filter_dim, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_1b = ResBlock(args, filters=filter_dim, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

        self.res_2a = ResBlock(args, filters=filter_dim, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_2b = ResBlock(args, filters=filter_dim, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

        self.res_3a = ResBlock(args, filters=2*filter_dim, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_3b = ResBlock(args, filters=2*filter_dim, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

        self.res_4a = ResBlock(args, filters=4*filter_dim, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_4b = ResBlock(args, filters=4*filter_dim, downsample=False, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

        # self.self_attn = Self_Attn(2 * filter_dim, self.act)

        # self.energy_map = nn.Linear(filter_dim*8, 1)


    def forward(self, x, compute_feat=False):
        x = self.act(self.conv1(x))

        x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_1a(x)
        x = self.res_1b(x)

        x = self.res_2a(x)
        x = self.res_2b(x)

        # if self.args.self_attn:
        # x, _ = self.self_attn(x)

        x = self.res_3a(x)
        x = self.res_3b(x)

        x = self.res_4a(x)
        x = self.res_4b(x)
        # x = self.act(x)
        return x


