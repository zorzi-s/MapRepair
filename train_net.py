import os
import glob

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import click
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries
from skimage import io
import itertools

from network import R2U_Net

from training_utils import sample_images, LossBuffer
import variables as var
from alignment_loss import AlignLoss
from data_loader import DataLoader


def train(models_path='./saved_models/', batch_size=2, \
    start_epoch=1, epochs=500, n_batches=1000, start_lr=0.0001, save_sample=100):
    Tensor = torch.cuda.FloatTensor

    border = var.BORDER
    window_size = var.WS

    net = R2U_Net(img_ch=3+1, t=2)

    if var.LOAD_MODEL_WEIGHTS:
        net.load_state_dict(torch.load(var.MODEL))

    net = net.cuda()

    os.makedirs(models_path, exist_ok=True)
    
    loss_net_buffer = LossBuffer()
    loss_net_buffer1 = LossBuffer()
    loss_net_buffer2 = LossBuffer()
    loss_net_buffer3 = LossBuffer()
    loss_net_buffer4 = LossBuffer()

    gen_obj = DataLoader(bs=batch_size, nb=n_batches, ws=window_size)
    
    optimizer_G = optim.Adam(net.parameters(), lr=start_lr)

    align_criterion = AlignLoss(window_size=window_size, border=border)
    align_criterion = align_criterion.cuda()
    bce_criterion = nn.BCELoss()
    bce_criterion = bce_criterion.cuda()

    for epoch in range(start_epoch, epochs):
        loader = gen_obj.generator()
        train_iterator = tqdm(loader, total=n_batches + 1)
        net.train()

        for i, (rgb, gti, miss, mod, inj) in enumerate(train_iterator):
            mod_inj = np.logical_or(mod, inj)
            gti_miss = np.logical_or(gti, miss)

            rgb = Variable(Tensor(rgb))
            gti = Variable(Tensor(gti))
            miss = Variable(Tensor(miss))
            mod = Variable(Tensor(mod))
            inj = Variable(Tensor(inj))
            mod_inj = Variable(Tensor(mod_inj))
            gti_miss = Variable(Tensor(gti_miss))

            rgb = rgb.permute(0,3,1,2)
            gti = gti.permute(0,3,1,2)
            miss = miss.permute(0,3,1,2)
            mod = mod.permute(0,3,1,2)
            inj = inj.permute(0,3,1,2)
            mod_inj = mod_inj.permute(0,3,1,2)
            gti_miss = gti_miss.permute(0,3,1,2)

            # Train Generators
            optimizer_G.zero_grad()

            trs, rot, sca, seg, seg_miss, seg_inj = net(rgb, mod_inj)

            align_loss, proj = align_criterion(rgb, mod, gti, seg_inj, trs, rot, sca)
            seg_loss = bce_criterion(seg, gti_miss)
            miss_loss = bce_criterion(seg_miss, miss)
            inj_loss = bce_criterion(seg_inj, inj)

            net_loss = align_loss + seg_loss + miss_loss + inj_loss

            net_loss.backward()
            optimizer_G.step()

            status = "[Epoch: %d][loss_net: %2.4f][align: %2.4f, seg: %2.4f, miss: %2.4f, inj: %2.4f]" % (epoch, \
                    loss_net_buffer.push(net_loss.item()), \
                    loss_net_buffer1.push(align_loss.item()), \
                    loss_net_buffer2.push(seg_loss.item()), \
                    loss_net_buffer3.push(miss_loss.item()), \
                    loss_net_buffer4.push(inj_loss.item()), )
            train_iterator.set_description(status)

            if (i % save_sample == 0):
                mask = gti[:,0,:,:].unsqueeze(1)
                mask = torch.cat((mask, mask), dim=1)
                #rgb[:,:,border,:] = 1
                #rgb[:,:,-border,:] = 1
                #rgb[:,:,:,border] = 1
                #rgb[:,:,:,-border] = 1
                sample_images(i, rgb, trs, [gti, mod_inj, proj, seg_miss, seg_inj])
        torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["alignNet", str(epoch)])))

        
if __name__ == '__main__':
    train()
