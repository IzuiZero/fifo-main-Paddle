import paddle
import paddle.nn as nn


def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2D(in_planes, affine=True, epsilon=1e-5, momentum=0.1)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2D(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias_attr=bias
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2D(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias_attr=bias
    )


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(
            nn.Conv2D(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=int(kernel_size / 2.0),
                groups=groups,
                bias_attr=False,
            ),
            batchnorm(out_planes),
            nn.ReLU6(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2D(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=int(kernel_size / 2.0),
                groups=groups,
                bias_attr=False,
            ),
            batchnorm(out_planes),
        )


class CRPBlock(nn.Layer):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(
                self,
                "{}_{}".format(i + 1, "outvar_dimred"),
                conv1x1(
                    in_planes if (i == 0) else out_planes,
                    out_planes,
                    stride=1,
                    bias=False,
                ),
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2D(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, "{}_{}".format(i + 1, "outvar_dimred"))(top)
            x = top + x
        return x
