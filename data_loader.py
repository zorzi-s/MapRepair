import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import random

from skimage import io
from skimage.segmentation import mark_boundaries
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage import measure
from skimage.transform import rotate, rescale

import variables as var


def to_categorical(y, num_classes=None, dtype='float32'):

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class DataLoader():

    def __init__(self, ws=512, nb=10000, bs=2):
        self.nb = nb
        self.bs = bs
        self.ws = ws

        self.load_data()
        self.num_tiles = len(self.rgb_imgs)
        self.sliding_index = 0


    def generator(self):
        for _ in range(self.nb):
            batch_rgb = []
            batch_gti = []
            batch_miss = []
            batch_mod = []
            batch_inj = []
            for _ in range(self.bs):
                rgb, gti, miss, mod, inj = self.extract_image()

                batch_rgb.append(rgb)

                # the ground truth is categorized
                #gti = to_categorical(gti != 0, 2)
                gti = np.expand_dims(gti, -1)
                batch_gti.append(gti)

                # the missing instances are categorized
                #miss = to_categorical(miss != 0, 2)
                miss = np.expand_dims(miss, -1)
                batch_miss.append(miss)

                # the segmentation is categorized
                #mod = to_categorical(mod != 0, 2)
                mod = np.expand_dims(mod, -1)
                batch_mod.append(mod)

                # the injections are categorized
                #inj = to_categorical(inj != 0, 2)
                inj = np.expand_dims(inj, -1)
                batch_inj.append(inj)

            batch_rgb = np.asarray(batch_rgb)
            batch_gti = np.asarray(batch_gti)
            batch_miss = np.asarray(batch_miss)
            batch_mod = np.asarray(batch_mod)
            batch_inj = np.asarray(batch_inj)

            batch_rgb = batch_rgb / 255.0

            yield (batch_rgb, batch_gti, batch_miss, batch_mod, batch_inj)


    def random_hsv(self, img, value_h=30, value_s=30, value_v=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        h = np.int16(h)
        s = np.int16(s)
        v = np.int16(v)

        h += value_h
        h[h < 0] = 0
        h[h > 255] = 255

        s += value_s
        s[s < 0] = 0
        s[s > 255] = 255

        v += value_v
        v[v < 0] = 0
        v[v > 255] = 255

        h = np.uint8(h)
        s = np.uint8(s)
        v = np.uint8(v)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img


    def translate_mask(self, mask, max_t=32):
        #mi = np.round(random.gauss(0, 0.2) * max_t)
        #mj = np.round(random.gauss(0, 0.2) * max_t)

        mi = np.random.randint(-max_t, max_t)
        mj = np.random.randint(-max_t, max_t)

        tform = SimilarityTransform(translation=(mi, mj))
        mask = warp(mask, tform, preserve_range=True)
        mask = np.uint8(mask!=0)
        return mask


    def globalFixedTranslation(self, mask, max_t=16):

        mi = max_t / np.sqrt(2)
        mj = max_t / np.sqrt(2)

        tform = SimilarityTransform(translation=(mi, mj))
        mask = warp(mask, tform, preserve_range=True)
        mask = np.uint8(mask!=0)
        return mask


    def rotate_mask(self, mask, max_a=30):
        angle = np.round(random.gauss(0, 0.2) * max_a)
        indices = np.argwhere(mask != 0)
        if indices.shape[0] > 0:
            ci = int(np.mean(indices[:,0]))
            cj = int(np.mean(indices[:,1]))
            mask = rotate(mask, angle, False, (cj, ci), preserve_range=True)
        return mask


    def scale_mask(self, mask, max_s=0.1):
        scale = 1 + random.uniform(-max_s, max_s)
        indices = np.argwhere(mask != 0)
        if indices.shape[0] > 0:
            ci = int(np.mean(indices[:,0]))
            cj = int(np.mean(indices[:,1]))

            k = np.eye(3)
            k[2,2] = scale
            mask = warp(mask, k)
            indices = np.argwhere(mask != 0)
            if indices.shape[0] > 0:
                di = int(np.mean(indices[:,0]))
                dj = int(np.mean(indices[:,1]))
                indices[:,0] = indices[:,0] - di + ci
                indices[:,1] = indices[:,1] - dj + cj

                mask = mask * 0
                mask[indices[:,0], indices[:,1]] = 1
                mask = np.uint8(mask != 0)
        return mask


    #def misalign(self, gti_, p_glob_trs=1.1, p_trs=0.9, p_rot=0.9, p_sca=0.0):
    def misalign(self, gti_, p_glob_trs=0.75, p_trs=0.9, p_rot=0.9, p_sca=0.0):
        gti = np.copy(gti_)
        if random.uniform(0,1) < p_glob_trs:
            gti = self.translate_mask(gti, max_t=16)
            #gti = self.globalFixedTranslation(gti, max_t=16)

        conn = np.uint16(measure.label(gti, background=0))
        n_conn = np.amax(conn)

        missalignment = np.copy(gti)
        for ins in range(1, n_conn+1):
            ins_mask = conn == ins

            is_overlapped = True
            while is_overlapped:
                mod_mask = np.copy(ins_mask)

                if random.uniform(0,1) < p_trs:
                    mod_mask = self.translate_mask(mod_mask, max_t=32)
                if random.uniform(0,1) < p_rot:
                    mod_mask = self.rotate_mask(mod_mask)
                if random.uniform(0,1) < p_sca:
                    mod_mask = self.scale_mask(mod_mask)

                temp = np.copy(missalignment)
                temp[ins_mask!=0] = 0
                temp[mod_mask!=0] = 1
                new_n_conn = np.amax(np.uint16(measure.label(temp, background=0)))
                if new_n_conn == n_conn:
                    missalignment = temp
                    is_overlapped = False

        return missalignment



    def filter_injections(self, inj, mod, p_injection=0.20):
        conn = np.uint16(measure.label(inj, background=0))
        n_conn = np.amax(conn)

        for ins in range(1, n_conn+1):
            ins_mask = conn == ins
            if np.count_nonzero(mod[ins_mask]) > 0:
                inj[ins_mask] = 0 # instance discarded
            elif np.random.uniform(0,1) > p_injection:
                inj[ins_mask] = 0 # instance discarded

        return inj


    def filter_instances(self, gti, p_filtering=0.20):
        conn = np.uint16(measure.label(gti, background=0))
        n_conn = np.amax(conn)

        filtered = np.zeros(gti.shape)

        for ins in range(1, n_conn+1):
            if np.random.uniform(0,1) < p_filtering:
                gti[conn == ins] = 0
                filtered[conn == ins] = 1

        return gti, filtered



    def extract_image(self, glob_t=16, p_t=0.75):
        rand_injected = random.randint(0, self.num_tiles-1)
        if self.sliding_index < self.num_tiles:
            rand_t = self.sliding_index
            self.sliding_index = self.sliding_index + 1
        else:
            rand_t = 0
            self.sliding_index = 0

        rgb = self.rgb_imgs[rand_t].copy()
        gti = self.gti_imgs[rand_t].copy()
        inj = self.gti_imgs[rand_injected].copy() # image used for injections

        h = rgb.shape[1]
        w = rgb.shape[0]

        """
        Extract thumbnail and perform some data augmentation
        """
        void = True
        while void:
            rot = random.randint(0,90)
            ri = random.randint(0, int(h-self.ws*2))
            rj = random.randint(0, int(w-self.ws*2))
            win_rgb = rgb[ri:ri+int(self.ws*2), rj:rj+int(self.ws*2)]
            win_gti = gti[ri:ri+int(self.ws*2), rj:rj+int(self.ws*2)]
            
            win_rgb = np.uint8(rotate(win_rgb, rot, resize=False, preserve_range=True))
            win_gti = np.uint8(rotate(win_gti, rot, resize=False, preserve_range=True))
            
            win_rgb = win_rgb[self.ws//2:-self.ws//2, self.ws//2:-self.ws//2]
            win_gti = win_gti[self.ws//2:-self.ws//2, self.ws//2:-self.ws//2]
            
            # Perform some data augmentation
            rot = random.randint(0,3)
            win_rgb = np.rot90(win_rgb, k=rot)
            win_gti = np.rot90(win_gti, k=rot)
            if random.randint(0,1) is 1:
                win_rgb = np.fliplr(win_rgb)
                win_gti = np.fliplr(win_gti)

            r_h = random.randint(-20,20)
            r_s = random.randint(-20,20)
            r_v = random.randint(-20,20)
            win_rgb = self.random_hsv(win_rgb, r_h, r_s, r_v)

            win_gti = np.uint8(win_gti!=0)

            # Create gti and miss masks
            win_gti, win_miss = self.filter_instances(win_gti)

            win_mod = np.copy(win_gti)
            win_mod = self.misalign(win_mod)
            win_mod = np.uint8(win_mod!=0)

            if np.count_nonzero(win_gti) and np.count_nonzero(win_mod):
            	void = False



        """
        Extract a thumbnail from the injection source image
        """
        void = True
        while void:
            rot = random.randint(0,90)
            ri = random.randint(0, int(h-self.ws*2))
            rj = random.randint(0, int(w-self.ws*2))
            win_inj = inj[ri:ri+int(self.ws*2), rj:rj+int(self.ws*2)]
            win_inj = np.uint8(rotate(win_inj, rot, resize=False, preserve_range=True))
            win_inj = win_inj[self.ws//2:-self.ws//2, self.ws//2:-self.ws//2]
            
            # Perform some data augmentation
            rot = random.randint(0,3)
            win_inj = np.rot90(win_inj, k=rot)
            if random.randint(0,1) is 1:
                win_inj = np.fliplr(win_inj)

            win_inj = np.uint8(win_inj!=0)
            win_inj = self.misalign(win_inj)
            win_inj = np.uint8(win_inj!=0)

            if np.count_nonzero(win_inj):
            	void = False

        win_inj = self.filter_injections(win_inj, win_mod)

        win_rgb = win_rgb.astype(np.float32)

        win_gti = win_gti.astype(np.float32)
        win_miss = win_miss.astype(np.float32)

        win_mod = win_mod.astype(np.float32)
        win_inj = win_inj.astype(np.float32)
        return (win_rgb, win_gti, win_miss, win_mod, win_inj)

        
    def load_data(self):
        self.rgb_imgs = []
        self.gti_imgs = []
        self.seg_imgs = []

        rgb_files = glob(var.DATASET_RGB)
        gti_files = glob(var.DATASET_GTI)

        rgb_files.sort()
        gti_files.sort()

        print("RGB files: %d" % len(rgb_files))
        print("GTI files: %d" % len(gti_files))
        assert len(rgb_files) == len(gti_files)

        combined = list(zip(rgb_files, gti_files))
        random.shuffle(combined)

        rgb_files[:], gti_files[:] = zip(*combined)

        if var.LOAD_FEW_DATA_SAMPLES:
            rgb_files = rgb_files[:4]
            gti_files = gti_files[:4]

        for rgb_name, gti_name in tqdm(zip(rgb_files, gti_files), total=len(rgb_files), desc="Loading dataset into RAM"):

            tmp = io.imread(rgb_name)
            tmp = tmp.astype(np.uint8)
            self.rgb_imgs.append(tmp)

            tmp = io.imread(gti_name)
            tmp = tmp.astype(np.uint8)
            self.gti_imgs.append(tmp)


