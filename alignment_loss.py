import torch.nn as nn
import torch
import kornia as tgm
import numpy as np
import random
from torch.autograd import Variable

from skimage import io
from skimage import measure
from skimage import filters

from data_loader import DataLoader


class AlignLoss(torch.nn.Module):

    def __init__(self, window_size=512, border=32):
        super(AlignLoss, self).__init__()
        self.window_size = window_size
        self.border = border

        self.warper = tgm.HomographyWarper(window_size, window_size, normalized_coordinates=True, mode="nearest")

        self.L1_criterion = torch.nn.MSELoss()
        self.L1_criterion = self.L1_criterion.cuda()

        self.L2_criterion = torch.nn.L1Loss()
        self.L2_criterion = self.L2_criterion.cuda()


    def makeProjection(self, gti_b, inj_b, trs_b, rot_b, sca_b):
        Tensor = torch.cuda.FloatTensor
        n_batches = gti_b.shape[0]

        projection = Variable(Tensor(np.zeros((n_batches, self.window_size, self.window_size))))

        for batch in range(n_batches):

            conn = gti_b[batch].clone()
            conn = conn.cpu().numpy()
            conn = np.uint16(measure.label(conn, background=0))

            n_instances = np.amax(conn)

            trs = trs_b[batch]
            rot = rot_b[batch]
            sca = sca_b[batch]
            inj = inj_b[batch]
        
            for ins in range(1, n_instances+1):
                indices = np.argwhere(conn == ins)

                ins_mask = np.zeros((self.window_size, self.window_size))
                ins_mask[indices[:,0], indices[:,1]] = 1.0
                ins_mask = Variable(Tensor(ins_mask))

                # Compute center of mass
                bx = np.mean(indices[:,1])
                by = np.mean(indices[:,0])

                remove = torch.mean(inj[indices[:,0], indices[:,1]])
                if remove < 0.5: # if close to zero the instance is not to remove

                    ti = torch.mean(trs[0, indices[:,0], indices[:,1]])
                    tj = torch.mean(trs[1, indices[:,0], indices[:,1]])
                    r = torch.mean(rot[0, indices[:,0], indices[:,1]])
                    s = torch.mean(sca[0, indices[:,0], indices[:,1]])

                    # Computation of the homograpy
                    bx = ((self.window_size // 2) - bx) / (self.window_size // 2)
                    by = ((self.window_size // 2) - by) / (self.window_size // 2)

                    R = torch.eye(3,3)
                    R[0,0] = torch.cos(r)
                    R[0,1] = -torch.sin(r)
                    R[1,0] = torch.sin(r)
                    R[1,1] = torch.cos(r)

                    T = torch.eye(3,3)
                    T[0,2] = ti
                    T[1,2] = tj

                    S = torch.eye(3,3)
                    S[0,0] = 1 + s
                    S[1,1] = 1 + s

                    B = torch.eye(3,3)
                    B[0,2] = bx
                    B[1,2] = by

                    B_ = torch.eye(3,3)
                    B_[0,2] = -bx
                    B_[1,2] = -by

                    H = torch.mm(R,B)
                    H = torch.mm(S,H)
                    H = torch.mm(B_,H)
                    H = torch.mm(T,H)

                    H = H.inverse().cuda()
                    #H[0,0] = 1.0
                    #H[1,1] = 1.0
                    #H[2,2] = 1.0
                    #H[0,2] = ti
                    #H[1,2] = tj

                    ins_mask = self.warper(ins_mask.view(1,1,self.window_size,self.window_size), H.view(1,1,3,3))
                    ins_mask = ins_mask[0,0,:,:]

                    projection[batch] += ins_mask
        return projection


    def prepareData(self, rgb, mod, gti, seg_inj):
        assert rgb.shape[0] == mod.shape[0] == gti.shape[0] == seg_inj.shape[0]
        assert mod.shape[1] == gti.shape[1] == seg_inj.shape[1] == 1
        mod = mod[:,0,:,:]
        gti = gti[:,0,:,:]
        seg_inj = seg_inj[:,0,:,:]
        return rgb, mod, gti, seg_inj


    def forward(self, rgb, mod, gti, seg_inj, trs, rot, sca):
        rgb, mod, gti, seg_inj = self.prepareData(rgb, mod, gti, seg_inj)

        proj = self.makeProjection(mod, seg_inj, trs, rot, sca)

        b = self.border
        loss_L1 = self.L1_criterion(proj[:, b:-b, b:-b], gti[:, b:-b, b:-b])
        loss_L2 = self.L2_criterion(proj[:, b:-b, b:-b], gti[:, b:-b, b:-b])
        loss = loss_L1 + loss_L2

        proj = proj != 0
        proj = proj.unsqueeze(1)

        return loss, proj

