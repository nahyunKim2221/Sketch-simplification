import torch
import random

import os
import argparse
from trainer import sketchGAN

def prepare(opts):
    # don't need to prepare something in the test only case
    if opts.testmodel_root is not None and opts.testonly:
        return
    if opts.seed == 0:
        opts.seed = random.randint(1, 10000)
    
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    opts.logpath = os.path.join("./exp", opts.exp, str(opts.seed))
    opts.testimagepath = os.path.join(opts.logpath, "testimages")
    opts.model_save_path = os.path.join(opts.logpath, "saved_models")

    os.makedirs(opts.testimagepath, exist_ok=True)
    os.makedirs(opts.model_save_path, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pair',action='store_true', help='if true -> the 2nd stage')
    parser.add_argument('--dataroot_i', type=str, help='dataroot for input training dataset')
    parser.add_argument('--dataroot_t', type=str, help='dataroot for target training dataset')
    parser.add_argument('--dataroot_test', type=str, help='dataroot for test dataset')
    parser.add_argument('--testmodel_root', default=None, type=str, help='path of pretrained model to test')
    parser.add_argument('--pretrained_root', type=str, help='[the 2nd stage] path of the 1st stage model in the 1st stage')

    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--numworker',type=int, default=1)
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--exp', type=str, help='the name of the experiment')
    
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--testonly', action='store_true', help='only test without training')
    parser.add_argument('--corrputed_inputs', action='store_true', help='[the 2nd stage] corrupt inputs when test the model')
    
    parser.add_argument('--lrG',type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--lrD',type=float, default=0.0001, help='learning rate for D')

    parser.add_argument('--ad_w', type=float, default=1.0, help='adversarial loss weight')
    parser.add_argument('--vgg_w', type=float, default=0.5, help='[the 1st stage] VGG-19 loss weight')
    parser.add_argument('--contour_w', type=float, default=0.5, help='[the 2nd stage] contour loss weight')
    
    parser.add_argument('--decay_epoch', type=int, default=10, help='number of epochs for decaying lr')
    parser.add_argument('--epoch', type=int, default=20, help='number of total epochs for training')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')

    opts = parser.parse_args()
    prepare(opts)
    skGAN = sketchGAN(opts)
    
    if opts.testonly:
        if opts.testmodel_root is not None:
            
            opts.testimagepath = os.path.join(opts.testmodel_root, "testimages")
            opts.model_save_path = opts.testmodel_root
            os.makedirs(opts.model_save_path, exist_ok=True)

        skGAN.bulid_model(opts)
        skGAN.test(0, opts, test_only=True)

    else:
        skGAN.bulid_model(opts)
        skGAN.train(opts)
