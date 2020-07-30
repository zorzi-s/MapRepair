import os
import glob
import sys

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries
from skimage import io
import itertools
import gdal
from skimage import measure

import variables as var

from network import R2U_Net
from alignment_loss import AlignLoss
from models import GeneratorResNet, Encoder
from regularization_lib import regularization


def copyGeoreference(inp, output):
    dataset = gdal.Open(inp)
    if dataset is None:
        print('Unable to open', inp, 'for reading')
        sys.exit(1)

    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is None and geotransform is None:
        print('No projection or geotransform found on file' + input)
        sys.exit(1)

    dataset2 = gdal.Open(output, gdal.GA_Update)

    if dataset2 is None:
        print('Unable to open', output, 'for writing')
        sys.exit(1)

    if geotransform is not None and geotransform != (0, 1, 0, 0, 0, 1):
        dataset2.SetGeoTransform(geotransform)

    if projection is not None and projection != '':
        dataset2.SetProjection(projection)

    gcp_count = dataset.GetGCPCount()
    if gcp_count != 0:
        dataset2.SetGCPs(dataset.GetGCPs(), dataset.GetGCPProjection())

    dataset = None
    dataset2 = None


def predict(rgb, gti, model, stn):
    x = np.copy(rgb)
    xx = np.copy(gti)

    x = x[np.newaxis,:,:,:]
    xx = xx[np.newaxis,:,:,:]

    Tensor = torch.cuda.FloatTensor
    x = Variable(Tensor(x))
    xx = Variable(Tensor(xx))

    x = x.permute(0,3,1,2)
    xx = xx.permute(0,3,1,2)
    
    trs, rot, sca, seg, seg_miss, seg_inj = model(x, xx)
    _, proj = stn(x, xx, xx, seg_inj, trs, rot, sca)
    seg_miss = torch.round(seg_miss)

    proj = proj.permute(0,2,3,1)
    proj = proj.detach().cpu().numpy()
    proj = proj.squeeze()
    seg_miss = seg_miss.permute(0,2,3,1)
    seg_miss = seg_miss.detach().cpu().numpy()
    seg_miss = seg_miss.squeeze()
    return proj, seg_miss


def alignNetwork(rgb, gti, model, stn):
    assert rgb.shape[0] == gti.shape[0]
    assert rgb.shape[1] == gti.shape[1]
    height = rgb.shape[0]
    width = rgb.shape[1]
    
    b = var.BORDER
    window_size = var.WS - 2*b
    
    ri = (height % window_size)
    rj = (width % window_size)
    
    # Prepare new bordered tile
    RGB = np.full((height-ri+window_size+2*b, width-rj+window_size+2*b, 3),0.0)
    RGB[b:-(window_size + b - ri), b:-(window_size + b - rj), :] = rgb
    rgb = None

    GTI = np.full((height-ri+window_size+2*b, width-rj+window_size+2*b, 1),0.0)
    GTI[b:-(window_size + b - ri), b:-(window_size + b - rj), :] = gti[:,:,np.newaxis]
    gti = None
    
    # Prepare the evaluation result tile
    ALIGN = np.full((height-ri+window_size, width-rj+window_size, 1), 0.0)
    MISS = np.full((height-ri+window_size, width-rj+window_size, 1), 0.0)
    
    h = RGB.shape[0]
    w = RGB.shape[1]
    
    step = var.WS - 2 * b
    ci = 0
    while(ci + step < h):
            cj = 0
            while(cj + step < w):
                    mini_RGB = np.copy(RGB[ci:ci+var.WS, cj:cj+var.WS, :])
                    mini_GTI = np.copy(GTI[ci:ci+var.WS, cj:cj+var.WS, :])
                    mini_ALIGN, mini_MISS = predict(mini_RGB, mini_GTI, model, stn)
                    ALIGN[ci:ci+window_size, cj:cj+window_size, :] = mini_ALIGN[b:-b, b:-b, np.newaxis]
                    MISS[ci:ci+window_size, cj:cj+window_size, :] = mini_MISS[b:-b, b:-b, np.newaxis]
                    cj += step
            ci += step
    
    ALIGN = ALIGN[0:-(window_size-ri), 0:-(window_size-rj)]
    MISS = MISS[0:-(window_size-ri), 0:-(window_size-rj)]
    MISS[:10,:] = 0
    MISS[:,:10] = 0
    MISS[-10:,:] = 0
    MISS[:,-10:] = 0
    return ALIGN, MISS


def align_gti(rgb, gti, dir_model):

    net = R2U_Net(img_ch=3+1, t=2)
    net.load_state_dict(torch.load(dir_model))
    net = net.cuda()

    stn = AlignLoss(window_size=var.WS, border=var.BORDER)
    stn = stn.cuda()

    aligned, missing = alignNetwork(rgb, gti, net, stn)

    net = None
    stn = None

    E1 = Encoder()
    G = GeneratorResNet()
    G.load_state_dict(torch.load('./saved_models/regularization/E140000_net'))
    E1.load_state_dict(torch.load('./saved_models/regularization/E140000_e1'))
    E1 = E1.cuda()
    G = G.cuda()

    regnet = [E1,G]

    missing = np.uint16(measure.label(missing, background=0))
    missing = regularization(rgb*255, missing, regnet)
    missing = missing != 0

    E1 = None
    G = None

    return aligned, missing


def prepare_input_data(rgb, gti):
    assert len(rgb.shape) == 3
    if len(gti.shape) == 3:
        gti = gti[:,:,2]
    
    rgb = rgb.astype(np.float32)
    gti = gti.astype(np.float32)
    if np.amax(rgb) > 1:
        rgb = rgb / 255.0
    if np.amax(gti) > 1:
        gti = gti / 255.0
    
    return rgb, gti
    

def align(dataset_rgb=var.PREDICTION_RGB, dataset_gti=var.PREDICTION_GTI, out_folder=var.OUT_FOLDER, dir_model=var.PREDICTION_MODEL):

    rgb_files = glob.glob(dataset_rgb)
    gti_files = glob.glob(dataset_gti)
    rgb_files.sort()
    gti_files.sort()

    for rgb_filename, gti_filename in tqdm(zip(rgb_files, gti_files), total=len(rgb_files), desc="Prediction"):
        out_file = os.path.basename(rgb_filename)
        out_file = out_folder + out_file
        #out_file = os.path.splitext(out_file)[0]

        rgb = io.imread(rgb_filename)
        gti = io.imread(gti_filename)

        rgb, gti = prepare_input_data(rgb, gti)

        #for i in range(1):
        #    print("Iteration %d" % (i+1))
        aligned, missing = align_gti(rgb, gti, dir_model)
        aligned = aligned.squeeze()
        missing = missing.squeeze()
        #gti = aligned

        final = np.logical_or(aligned, missing)
        cv2.imwrite(out_file, np.uint8(final*255))
        copyGeoreference(rgb_filename, out_file)


if __name__ == '__main__':
    align()



