import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class LambdaLR:
    def __init__(self, n_eps, offset, decay_start_ep) -> None:
        assert (n_eps - decay_start_ep) > 0, 'Decay must start before the training session ends!'
        self.n_eps = n_eps
        self.offset = offset
        self.decay_start_ep = decay_start_ep
    
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_ep) / (self.n_eps - self.decay_start_ep)

class VGG_loss(nn.Module):
    def __init__(self, vgg):
        super(VGG_loss, self).__init__()
        self.vgg = vgg
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def forward(self, input, target):
        img_vgg = vgg_preprocess(input)
        target_vgg = vgg_preprocess(target)
        img_fea = self.vgg(img_vgg)
        target_fea = self.vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea))**2)

def vgg_preprocess(batch):
    tensortype = type(batch.data)
    if batch.size(1) == 1:
        batch = batch.repeat(1,3,1,1)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = batch * 255       #   * 0.5  [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))
    return batch

class ToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)