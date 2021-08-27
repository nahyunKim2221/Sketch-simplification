import torch
import torch.nn as nn

import torchvision.models as models


class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, middle_channel=64, n_blocks=6):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, middle_channel, kernel_size=3, stride=1, padding=1),
            InResBlock(middle_channel),
            InResBlock(middle_channel),
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(middle_channel, middle_channel*2, kernel_size=3, stride=1, padding=1),
            InResBlock(middle_channel*2),
            InResBlock(middle_channel*2),
        )

        conv3 = [nn.AvgPool2d(2), nn.Conv2d(middle_channel*2, middle_channel*4, kernel_size=3, stride=1, padding=1)]

        for i in range(n_blocks):
            conv3 += [InResBlock(middle_channel*4)]

        self.conv3 = nn.Sequential(*conv3)

        self.up1 = nn.ConvTranspose2d(middle_channel*4, middle_channel*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(middle_channel*2, middle_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Sequential(
            InResBlock(middle_channel*2),
            InResBlock(middle_channel*2)
        )

        self.conv5 = nn.Sequential(
            InResBlock(middle_channel),
            InResBlock(middle_channel),
            nn.Conv2d(middle_channel, output_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(self.up1(conv3) + conv2)
        conv5 = self.conv5(self.up2(conv4) + conv1)
        out = conv5 + x
        return out



class InResBlock(nn.Module):
    def __init__(self, dim):
        super(InResBlock, self).__init__()

        self.norm1 = IN(dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0)

        self.norm2 = IN(dim)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.pad1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.pad2(out)
        out = self.conv2(out)

        out = out + x

        return out


class IN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(IN, self).__init__()

        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.betta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(
            x, dim=[2, 3], keepdim=True
        )

        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)

        out = self.gamma.expand(x.shape[0], -1, -1, -1) * out_in + self.betta.expand(
            x.shape[0], -1, -1, -1
        )

        return out

class Discriminator(nn.Module):
    ''' below codes are borrowed from https://github.com/yunjey/stargan/blob/master/model.py '''
    def __init__(self, inpt_ch, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(inpt_ch, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src


class vgg_19(nn.Module):
    def __init__(self):
        super(vgg_19, self).__init__()
        vgg_model = models.vgg19(pretrained=True)
        
        #self.feature_ext = nn.Sequential(*list(vgg_model.features.children())[:20])
        self.feature_ext = nn.Sequential(*list(vgg_model.features.children())[:27]) # more abstracted feature vgg 5-4
    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat((x, x, x), 1)
        out = self.feature_ext(x)
        return out
