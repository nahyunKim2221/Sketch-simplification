from PIL import Image
import os
import os.path
from natsort import natsorted

import torch
import torch.utils.data as data
from torchvision.transforms import transforms 
import torchvision.transforms.functional as TF

import random
import numpy as np
import cv2

###############################################################################
# Code from
# https://github.com/Wenchao-Du/LIR-for-Unsupervised-IR/blob/master/data.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def default_loader(path):
    return Image.open(path).convert('L')


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                
    return images


class dataset_test(data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader, return_path=False):
        imgs = make_dataset(root)
        imgs = natsorted(imgs)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.need_path = return_path

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.need_path:
            return img, path
        else:
            return img

    def __len__(self):
        return min(len(self.imgs), 500) # 500 is the upper bound of test images to save the test time

    
class dataset_pair(data.Dataset):
    def __init__(self, input_root, target_root, loader=default_loader):
        input_imgs = natsorted(make_dataset(input_root))
        target_imgs = natsorted(make_dataset(target_root))
        if len(input_imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + input_root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        if len(target_imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + target_root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        assert len(input_imgs) == len(target_imgs), "# of input and gt must be same"

        self.input_imgs = input_imgs
        self.target_imgs = target_imgs

        self.loader = loader
        self.er_kernel = np.ones((3, 3), np.uint8)

    def __getitem__(self, index):
        input_path = self.input_imgs[index]
        target_path = self.target_imgs[index]

        input_img = self.loader(input_path)
        gt = self.loader(target_path)
        
        # find contour of gt
        gt_contour = cv2.resize(np.array(gt.copy()), dsize=(256, 256))
        contour_bg = np.ones_like(gt_contour) * 255
        thresh = cv2.threshold(gt_contour, 120, 255, cv2.THRESH_BINARY_INV)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
        contours = cnts[0] if len(cnts) == 2 else cnts[1]
        gt_contour = cv2.drawContours(contour_bg, contours, -1, (36, 255, 12), 2)
        
        
        input_img, gt, gt_contour = self.transform(input_img, gt, Image.fromarray(gt_contour))
        
        return input_img, gt, gt_contour
        
    def __len__(self):
        return len(self.input_imgs)
    
    def transform(self, image1, image2, image3):
        resize = transforms.Resize(size=(256, 256))
        image1 = resize(image1)
        image2 = resize(image2)
        image3 = resize(image3)

        # Random horizontal flipping
        if random.random() > 0.5:
            image1 = TF.hflip(image1)
            image2 = TF.hflip(image2)
            image3 = TF.hflip(image3)

        # Random vertical flipping
        if random.random() > 0.5:
            image1 = TF.vflip(image1)
            image2 = TF.vflip(image2)
            image3 = TF.vflip(image3)

        # Transform to tensor
        if not torch.is_tensor(image1):
            image1 = TF.to_tensor(image1)
        image2 = TF.to_tensor(image2)
        image3 = TF.to_tensor(image3)

        return image1, image2, image3
    
        
class dataset_train(data.Dataset):
    def __init__(self, input_root, target_root, transform=None, loader=default_loader):
        input_imgs = natsorted(make_dataset(input_root))
        target_imgs = natsorted(make_dataset(target_root))
        if len(input_imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + input_root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        if len(target_imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + input_root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.input_root = input_root
        self.target_root = target_root

        self.input_imgs = input_imgs
        self.target_imgs = target_imgs

        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        input_path = self.input_imgs[index]
        target_path = self.target_imgs[index]
        
        input_img = self.loader(input_path)
        target_img = self.loader(target_path)
        
        if self.transform is not None:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img
        
    def __len__(self):
        return min(len(self.input_imgs), len(self.target_imgs))
    