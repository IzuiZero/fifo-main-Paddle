import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Sequential
from utils.helpers import maybe_download
from utils.layer_factory import conv1x1, conv3x3, CRPBlock

data_info = {7: "Person", 21: "VOC", 40: "NYU", 60: "Context"}

models_urls = {
    "50_person": "https://cloudstor.aarnet.edu.au/plus/s/mLA7NxVSPjNL7Oo/download",
    "101_person": "https://cloudstor.aarnet.edu.au/plus/s/f1tGGpwdCnYS3xu/download",
    "152_person": "https://cloudstor.aarnet.edu.au/plus/s/Ql64rWqiTvWGAA0/download",
    "50_voc": "https://cloudstor.aarnet.edu.au/plus/s/xp7GcVKC0GbxhTv/download",
    "101_voc": "https://cloudstor.aarnet.edu.au/plus/s/CPRKWiaCIDRdOwF/download",
    "152_voc": "https://cloudstor.aarnet.edu.au/plus/s/2w8bFOd45JtPqbD/download",
    "50_nyu": "https://cloudstor.aarnet.edu.au/plus/s/gE8dnQmHr9svpfu/download",
    "101_nyu": "https://cloudstor.aarnet.edu.au/plus/s/VnsaSUHNZkuIqeB/download",
    "152_nyu": "https://cloudstor.aarnet.edu.au/plus/s/EkPQzB2KtrrDnKf/download",
    "101_context": "https://cloudstor.aarnet.edu.au/plus/s/hqmplxWOBbOYYjN/download",
    "152_context": "https://cloudstor.aarnet.edu.au/plus/s/O84NszlYlsu00fW/download",
    "50_imagenet": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "101_imagenet": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "152_imagenet": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

stages_suffixes = {0: "_conv", 1: "_conv_relu_varout_dimred"}


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False
        )
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetLW(nn.Layer):
    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(ResNetLW, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)

        self.clf_conv = nn.Conv2D(
            256, num_classes, kernel_size=3, stride=1, padding=1, bias_attr=True
        )

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        out1 = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        out2 = l1
        l2 = self.layer2(l1)
        out3 = l2
        l3 = self.layer3(l2)
        out4 = l3
        l4 = self.layer4(l3)
        out5 = l4

        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = F.interpolate(x4, size=l3.shape[2:], mode="bilinear", align_corners=True)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = F.interpolate(x3, size=l2.shape[2:], mode="bilinear", align_corners=True)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = F.interpolate(x2, size=l1.shape[2:], mode="bilinear", align_corners=True)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        out = self.clf_conv(x1)
        out6 = out

        return out1, out2, out3, out4, out5, out6

    def get_1x_lr_params_NOscale(self):
        b = []
        b.append(self.conv1.parameters())
        b.append(self.bn1.parameters())
        b.append(self.layer1.parameters())
        b.append(self.layer2.parameters())
        b.append(self.layer3.parameters())
        b.append(self.layer4.parameters())

        for i in range(len(b)):
            for j in b[i]:
                for k in j:
                    if k.trainable:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.clf_conv.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def rf_lw50(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = ResNetLW(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = "50_imagenet"
        url = models_urls[key]
        # PaddlePaddle's model loading is different, handle it according to PaddlePaddle's API
        # model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = "50_" + dataset.lower()
            key = "rf_lw" + bname
            url = models_urls[bname]
            # PaddlePaddle's model loading is different, handle it according to PaddlePaddle's API
            # model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def rf_lw101(num_classes, imagenet=True, pretrained=False, **kwargs):
    model = ResNetLW(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = "101_imagenet"
        url = models_urls[key]
        # PaddlePaddle's model loading is different, handle it according to PaddlePaddle's API
        # model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = "101_" + dataset.lower()
            key = "rf_lw" + bname
            url = models_urls[bname]
            # PaddlePaddle's model loading is different, handle it according to PaddlePaddle's API
            # model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def rf_lw152(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = ResNetLW(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = "152_imagenet"
        url = models_urls[key]
        # PaddlePaddle's model loading is different, handle it according to PaddlePaddle's API
        # model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = "152_" + dataset.lower()
            key = "rf_lw" + bname
            url = models_urls[bname]
            # PaddlePaddle's model loading is different, handle it according to PaddlePaddle's API
            # model.load_state_dict(maybe_download(key, url), strict=False)
    return model
