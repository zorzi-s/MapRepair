import numpy as np
import cv2
import glob
from tqdm import tqdm
import random
from skimage import io
from skimage.segmentation import mark_boundaries
import os

import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

import math

import variables as var


def flow2rgb(flow, max_value=None):

    flow = flow.detach().cpu().numpy()

    rgb = np.zeros((flow.shape[1], flow.shape[2], 3))

    x = np.abs(flow[0])
    y = np.abs(flow[1])
    abs_value = np.sqrt(np.power(x, 2) + np.power(y, 2))

    r, g, b = x, y, abs_value

    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b

    return rgb


def sample_images(sample_index, img, flow, masks, debug=True):
    os.makedirs("./debug/", exist_ok=True)
    batch = img.shape[0]

    img = img.permute(0,2,3,1)

    for i in range(len(masks)):
        masks[i] = masks[i].permute(0,2,3,1)

    img = img.cpu().numpy()
    ip = np.uint8(img * 255)
    for i in range(len(masks)):
        masks[i] = masks[i].detach().cpu().numpy()
        #masks[i] = np.argmax(masks[i], axis=-1)
        masks[i] = np.round(masks[i])
        masks[i] = np.uint8(masks[i] * 255)

    if debug:
        for i, m in enumerate(masks):
            io.imsave("./debug/debug_%d_mask%d.png" % (sample_index, i), m[0,:,:,0])
            io.imsave("./debug/debug_%d_img.png" % sample_index, np.uint8(255*img[0]))

    line_mode = "inner"
    for i in range(len(masks)):
        row = np.copy(ip[0,:,:,:])
        line = cv2.Canny(masks[i][0,:,:], 0, 255)
        row = mark_boundaries(row, line, color=(1,1,0), mode=line_mode) * 255#, outline_color=(self.red,self.greed,0))
        for b in range(1,batch):
            pic = np.copy(ip[b,:,:,:])
            line = cv2.Canny(masks[i][b,:,:], 0, 255)
            pic = mark_boundaries(pic, line, color=(1,1,0), mode=line_mode) * 255#, outline_color=(self.red,self.greed,0))
            row = np.concatenate((row, pic), 1)
        masks[i] = row


    flow = (flow - torch.min(flow)) / (torch.max(flow) - torch.min(flow))
    row = flow2rgb(flow[0,:,:,:])
    if debug:
        io.imsave("./debug/debug_%d_flow.png" % sample_index, np.uint8(255*row))
    for i in range(1,batch):
        row = np.concatenate((row, flow2rgb(flow[i,:,:])), axis=1)
    masks.append(row*255)


    img = np.concatenate(masks, 0)
    img = np.rot90(img)
    img = np.uint8(img)
    io.imsave(var.DEBUG_FOLDER + "debug_%s.png" % str(sample_index), img)


class LossBuffer():
    def __init__(self, max_size=100):
        self.data = []
        self.max_size = max_size

    def push(self, data):
        self.data.append(data)    
        if len(self.data) > self.max_size:
            self.data = self.data[1:]
        return sum(self.data) / len(self.data)


class LambdaLR():
    def __init__(self, n_batches, decay_start_batch):
        assert ((n_batches - decay_start_batch) > 0), "Decay must start before the training session ends!"
        self.n_batches = n_batches
        self.decay_start_batch = decay_start_batch

    def step(self, batch):
        if batch > self.decay_start_batch:
            factor = 1.0 - (batch - self.decay_start_batch) / (self.n_batches - self.decay_start_batch)
            if factor > 0:
                return factor
            else:
                return 0.0
        else:
            return 1.0

