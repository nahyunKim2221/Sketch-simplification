import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

from data_loader import dataset_test, dataset_train, dataset_pair
from utils import *
from models import Generator, Discriminator, vgg_19

import os
import sys

class sketchGAN(object):
    def __init__(self, opts):
        print(opts)
    
    def bulid_model(self, opts):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        """ transform & Dataloader """
        train_transform = transforms.Compose([
            transforms.Resize((opts.image_size, opts.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            ToNumpy(),
            transforms.ToTensor(),
        ])
        
        test_transform = transforms.Compose([transforms.Resize((opts.image_size, opts.image_size)), ToNumpy(), transforms.ToTensor()])
        
        if not opts.testonly:
            self.train_dataset = dataset_train(opts.dataroot_i, opts.dataroot_t, transform=train_transform)
            self.train_loader = DataLoader(self.train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.numworker)

        if opts.pair:
            self.train_dataset = dataset_pair(opts.dataroot_i, opts.dataroot_t)
            self.train_loader = DataLoader(self.train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.numworker)
            
            self.pretrained_g = Generator(1, 1, middle_channel=32, n_blocks=6)
            self.pretrained_g = nn.DataParallel(self.pretrained_g)
            self.pretrained_g.load_state_dict(torch.load(opts.pretrained_root)["geni2t"])
            self.pretrained_g.eval().to(self.device)
            # in pair training
            self.contour_Loss = nn.L1Loss().to(self.device)

                
        self.test_datset = dataset_test(opts.dataroot_test, transform=test_transform, return_path=False)
        self.test_loader = DataLoader(self.test_datset, batch_size=1, num_workers=opts.numworker)

        self.geni2t = Generator(1, 1, middle_channel=32, n_blocks=6)
        
        self.discriminator = Discriminator(1)

        self.geni2t = nn.DataParallel(self.geni2t)
        self.discriminator = nn.DataParallel(self.discriminator)
        
        # cycle loss for stable training
        self.recon_Loss = nn.L1Loss().to(self.device)
        # for LSGAN loss
        self.MSE_Loss = nn.MSELoss().to(self.device)
        # VGG loss
        vgg = vgg_19()

        if torch.cuda.is_available():
            vgg.to(self.device)
        
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False
        
        self.VGG_Loss = VGG_loss(vgg).to(self.device)        

        """ Optimizer """
        self.G_optim = optim.Adam(
            self.geni2t.parameters(),
            lr=opts.lrG,
        )
        self.D_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=opts.lrD,
        )
        opts.ckpt_path = os.path.join(opts.model_save_path, "last.pth")

    
    def train(self, opts):
        self.geni2t.train().to(self.device)
        self.discriminator.train().to(self.device)

        G_schedular = torch.optim.lr_scheduler.LambdaLR(
            self.G_optim, lr_lambda = LambdaLR(opts.epoch,0,opts.decay_epoch).step
        )
        D_schedular = torch.optim.lr_scheduler.LambdaLR(
            self.D_optim, lr_lambda = LambdaLR(opts.epoch,0,opts.decay_epoch).step
        )

        start = 0
        if opts.resume:
            ckpt = torch.load(opts.ckpt_path)
            self.geni2t.load_state_dict(ckpt["geni2t"])
            self.G_optim.load_state_dict(ckpt["G_optimizer"])
            start = ckpt["epoch"]
            if not opts.pair:
                self.discriminator.load_state_dict(ckpt["dis"])
                self.D_optim.load_state_dict(ckpt["D_optimizer"])
        

        if not opts.pair: # the 1st stage
            for epoch in range(start, opts.epoch):
                for i, (inputs, target) in enumerate(self.train_loader):
                    input = inputs.to(self.device)
                    target = target.to(self.device)
                    
                    # Update D
                    self.D_optim.zero_grad()

                    fake_i2t = self.geni2t(input)
                    fake_logit = self.discriminator(fake_i2t.detach())
                    real_logit = self.discriminator(target)
                    d_loss = self.MSE_Loss(real_logit, torch.ones_like(real_logit)) + self.MSE_Loss(fake_logit, torch.zeros_like(fake_logit))
                    
                    d_loss.backward()
                    self.D_optim.step()
                    
                    # Update G
                    self.G_optim.zero_grad()
                    
                    fake_logit = self.discriminator(fake_i2t)
                    
                    g_loss = opts.ad_w * self.MSE_Loss(fake_logit, torch.ones_like(fake_logit))
                    g_loss += opts.vgg_w * self.VGG_Loss(fake_i2t, input) # to preserve semantic info
                    
                    sys.stdout.write(f"\r[Epoch: {epoch}/{opts.epoch}] [Batch: {i:04d}/{len(self.train_loader)}] [G loss: {g_loss:.4f}] [D loss: {d_loss:.4f}]")
                    g_loss.backward()
                    self.G_optim.step()

                self.test(epoch, opts)
                G_schedular.step()
                D_schedular.step()
                ckpt = {
                    "geni2t": self.geni2t.state_dict(),
                    "dis": self.discriminator.state_dict(),
                    "G_optimizer": self.G_optim.state_dict(),
                    "D_optimizer": self.D_optim.state_dict(),
                    "seed": opts.seed,
                    "epoch":epoch,
                }
                
                torch.save(ckpt, os.path.join(opts.model_save_path, f"{epoch}.pth"))
            torch.save(ckpt, opts.ckpt_path)
       
        else: # the 2nd stage
            for epoch in range(start, opts.epoch):
                for i, (inputs, gt, gt_contour) in enumerate(self.train_loader):
                    input = inputs.to(self.device)
                    gt = gt.to(self.device)
                    gt_contour = gt_contour.to(self.device)
                    corrupted_input = self.pretrained_g(input).detach()
                    mask = 1.0 - gt_contour # reverse the contour to make masks
                    
                    # Update G
                    self.G_optim.zero_grad()

                    fake_i2t = self.geni2t(corrupted_input)

                    g_loss = self.recon_Loss(fake_i2t, gt)
                    g_loss += opts.contour_w * (1 - (epoch/opts.epoch)) * self.contour_Loss(mask*fake_i2t, mask*gt_contour)
                    
                    sys.stdout.write(f"\r[Epoch: {epoch}/{opts.epoch}] [Batch: {i:04d}/{len(self.train_loader)}] [G loss: {g_loss:.4f}]")

                    g_loss.backward()
                    self.G_optim.step()

                self.test(epoch, opts)
                G_schedular.step()
                
                ckpt = {
                    "geni2t": self.geni2t.state_dict(),
                    "G_optimizer": self.G_optim.state_dict(),
                    "seed": opts.seed,
                    "epoch":epoch,
                }
                
                torch.save(ckpt, os.path.join(opts.model_save_path, f"{epoch}.pth")) # .pth files save every epoch
                torch.save(ckpt, opts.ckpt_path)
                
    
    def test(self, epoch, opts, test_only=False):
        if test_only:
            print("start test!")
            self.test_datset = dataset_test(opts.dataroot_test, transform=transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()]), return_path=True)
            self.test_loader = DataLoader(self.test_datset, batch_size=1, num_workers=opts.numworker)

            ckpt = torch.load(opts.ckpt_path)
            self.geni2t.load_state_dict(ckpt["geni2t"])
            name = os.path.basename(os.path.normpath(opts.dataroot_test))
            epoch = f"testonly_{name}_ep" + str(ckpt["epoch"])
            
            with torch.no_grad():
                for i, (input, path) in enumerate(self.test_loader):
                    margin_2 = 0
                    margin_3 = 0
                    _, _, h, w = input.size()
                    if input.size(2) % 4 != 0:
                        margin_2 = input.size(2) % 4
                    if input.size(3) % 4 != 0:
                        margin_3 = input.size(3) % 4
                    if margin_2 + margin_3 != 0:
                        input = F.interpolate(input, (h - margin_2, w - margin_3))
                    input.to(self.device)
                    
                    if opts.pair and opts.corrputed_inputs:
                        ckpt_1st = torch.load(opts.pretrained_root)
                        self.pretrained_g.load_state_dict(ckpt_1st["geni2t"])
                        self.pretrained_g.eval().to(self.device)

                        input = self.pretrained_g(input)

                    output = self.geni2t(input)
                    output = torch.clamp(output, 0, 1)
                    
                    os.makedirs(f"{opts.testimagepath}/{epoch}", exist_ok=True)
                    os.makedirs(f"{opts.testimagepath}/input_{name}", exist_ok=True)
                    
                    img_name = os.path.basename(path[0])
                                        
                    save_image(input[0],f"{opts.testimagepath}/input_{name}/input_{img_name}", normalize=True)
                    save_image(output[0], f"{opts.testimagepath}/{epoch}/{img_name}", normalize=True)
                    

        else:
            with torch.no_grad():
                for i, input in enumerate(self.test_loader):
                            
                    margin_2 = 0
                    margin_3 = 0
                    _, _, h, w = input.size()
                    if input.size(2) % 4 != 0:
                        margin_2 = input.size(2) % 4
                    if input.size(3) % 4 != 0:
                        margin_3 = input.size(3) % 4
                    if margin_2 + margin_3 != 0:
                        input = F.interpolate(input, (h - margin_2, w - margin_3))
                    input.to(self.device)

                    output = self.geni2t(input)
                    output = torch.clamp(output, 0, 1)
                    os.makedirs(f"{opts.testimagepath}/{epoch}", exist_ok=True)
                    if epoch == 0:
                        os.makedirs(f"{opts.testimagepath}/input", exist_ok=True)
                        save_image(input[0],f"{opts.testimagepath}/input/input_{i}.png", normalize=True)
                    save_image(output[0], f"{opts.testimagepath}/{epoch}/output_{i}.png", normalize=True)
