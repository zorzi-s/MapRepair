import random
from skimage import io
from skimage.transform import rotate
import numpy as np
import torch
from tqdm import tqdm

from skimage import measure
import cv2

from models import GeneratorResNet, Encoder


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


def predict_building(rgb, mask, model):
	Tensor = torch.cuda.FloatTensor

	mask = to_categorical(mask, 2)

	rgb = rgb[np.newaxis, :, :, :]
	mask = mask[np.newaxis, :, :, :]

	E, G = model

	rgb = Tensor(rgb)
	mask = Tensor(mask)
	rgb = rgb.permute(0,3,1,2)
	mask = mask.permute(0,3,1,2)

	rgb = rgb / 255.0

	# PREDICTION
	pred = G(E([rgb, mask]))
	pred = pred.permute(0,2,3,1)

	pred = pred.detach().cpu().numpy()

	pred = np.argmax(pred[0,:,:,:], axis=-1)
	return pred



def fix_limits(i_min, i_max, j_min, j_max):

	def closest_divisible_size(size, factor=4):
		while size % factor:
			size += 1
		return size

	min_image_size = 256
	height = i_max - i_min
	width = j_max - j_min

	# pad the rows
	if height < min_image_size:
		diff = min_image_size - height
	else:
		diff = closest_divisible_size(height) - height + 16

	i_min -= (diff // 2)
	i_max += (diff // 2 + diff % 2)

	# pad the columns
	if width < min_image_size:
		diff = min_image_size - width
	else:
		diff = closest_divisible_size(width) - width + 16

	j_min -= (diff // 2)
	j_max += (diff // 2 + diff % 2)

	return i_min, i_max, j_min, j_max



def regularization(rgb, ins_segmentation, model):
	max_instance = np.amax(ins_segmentation)
	border = 256

	ins_segmentation = np.uint16(cv2.copyMakeBorder(ins_segmentation,border,border,border,border,cv2.BORDER_CONSTANT,value=0))
	rgb = np.uint8(cv2.copyMakeBorder(rgb,border,border,border,border,cv2.BORDER_CONSTANT,value=(0,0,0)))

	regularization = np.zeros(ins_segmentation.shape, dtype=np.uint16)

	for ins in tqdm(range(1, max_instance+1), desc="Regularization"):
		indices = np.argwhere(ins_segmentation==ins)
		building_size = indices.shape[0]
		if building_size > 0:
			i_min = np.amin(indices[:,0])
			i_max = np.amax(indices[:,0])
			j_min = np.amin(indices[:,1])
			j_max = np.amax(indices[:,1])

			i_min, i_max, j_min, j_max = fix_limits(i_min, i_max, j_min, j_max)

			mask = np.copy(ins_segmentation[i_min:i_max, j_min:j_max] == ins)
			rgb_mask = np.copy(rgb[i_min:i_max, j_min:j_max, :])

			pred = predict_building(rgb_mask, mask, model)

			pred_indices = np.argwhere(pred != 0)

			if pred_indices.shape[0] > 0:
				pred_indices[:,0] = pred_indices[:,0] + i_min
				pred_indices[:,1] = pred_indices[:,1] + j_min
				x, y = zip(*pred_indices)
				regularization[x,y] = ins

	return regularization[border:-border, border:-border]



