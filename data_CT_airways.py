import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import random
from functools import lru_cache
from utils import load_itk_image
from data_BAS import random_sample, central_crop, augment_random_rotate

np.random.seed(777) #numpy


class AirwayData(Dataset):
	"""
	Generate dataloader
	"""
	def __init__(self, config, phase='train', stage=1, split_comber=None,
				debug=False, crop_size=[128, 128, 128], random_select=False, small_airway=False):
		"""
		:param config: configuration from model
		:param phase: training or validation or testing
		:param split_comber: split-combination-er
		:param debug: debug mode to check few data
		:param random_select: use partly, randomly chosen data for training
		"""
		assert(phase == 'train' or phase == 'val' or phase == 'test')
		self.phase = phase
		self.augtype = config['augtype']
		self.split_comber = split_comber
		self.patch_per_case = 16  # patches used per case 
		self.debug_flag = debug
		self.stage = stage
		self.crop_size = crop_size

		"""
		specify the path and data split
		"""
		self.datapath = config['dataset_path']

        # Determine paths based on phase
        sub_folder = 'train' if phase == 'train' else ('val' if phase == 'val' else 'test')
        self.base_path = os.path.join(self.datapath, 'image_clean', sub_folder)
        
        self.data_file_names = sorted(os.listdir(self.base_path))
        if self.debug_flag:
            self.data_file_names = self.data_file_names[:1]
        
        self.caseNumber = len(self.data_file_names)
        
        # For Val/Test, we still need the cubelist for splitting 3D volumes
        self.cubelist = []
        if self.phase != 'train':
            print(f"--- Preparing {self.phase} metadata (No full load) ---")
            for file_name in self.data_file_names:
                raw_path = os.path.join(self.base_path, file_name)
                # We only load briefly to get metadata for splitting
                imgs, _, _ = load_itk_image(raw_path)
                splits, nzhw, orgshape = self.split_comber.split_id(imgs)
                data_name = file_name.split('.nii')[0]
                for j in range(len(splits)):
                    self.cubelist.append([data_name, splits[j], j, nzhw, orgshape, 'N'])
            
            if self.phase == 'val':
                random.shuffle(self.cubelist)

        print(f"Initialized {phase} with {self.caseNumber} cases.")

    # We cache the last 1-2 loaded volumes to avoid reloading 16 times per case
    @lru_cache(maxsize=2)
    def _load_data_from_disk(self, data_name):
        # Reconstruct paths
        raw_path = os.path.join(self.base_path, data_name + '.nii.gz') # or .nii
        
        if self.small_airway:
            label_path = raw_path.replace('image_clean', 'smallairway_clean').replace('clean_hu', 'smallairway')
            weight_path = raw_path.replace('image_clean', 'LIB_weight_small').replace('clean_hu.nii.gz', 'smallweight.npy')
        else:
            label_path = raw_path.replace('image_clean', 'label').replace('clean_hu', 'label')
            weight_path = raw_path.replace('image_clean', 'LIB_weight').replace('clean_hu.nii.gz', 'weight.npy')

        imgs, origin, spacing = load_itk_image(raw_path)
        labels, _, _ = load_itk_image(label_path)
        
        weight = None
        if self.phase == 'train':
            weight = np.load(weight_path)
            
        return imgs, labels, weight, origin, spacing

    def __len__(self):
        return self.patch_per_case * self.caseNumber if self.phase == 'train' else len(self.cubelist)

    def __getitem__(self, idx):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))

        if self.phase == 'train':
            data_name = self.data_file_names[idx // self.patch_per_case].split('.nii')[0]
            curNameID = data_name
            curtransFlag = 'Y'
        else:
            curlist = self.cubelist[idx]
            curNameID, cursplit, curSplitID, curnzhw, curShapeOrg, curtransFlag = curlist

        # LOAD HERE
        imgs_full, label_full, weight_full, origin, spacing = self._load_data_from_disk(curNameID)
        
        # Copy to avoid mutating cached data
        label = (label_full > 0).astype('float')
        imgs = imgs_full.copy()

        if self.phase != 'train':
            # Slice the specific cube for validation/testing
            z, y, x = cursplit
            label = label[z[0]:z[1], y[0]:y[1], x[0]:x[1]]
            imgs = imgs[z[0]:z[1], y[0]:y[1], x[0]:x[1]]
            imgs = (imgs.astype(np.float32)) / 255.0
        else:
            weight = weight_full.copy()
            # Random sampling/augmentation logic
            if imgs.shape[0] <= 145 or imgs.shape[1] <= 145 or imgs.shape[2] <= 145:
                imgs, label, weight = random_sample(imgs, label, weight, [128,128,128])
            else:
                imgs, label, weight = random_sample(imgs, label, weight, [145,145,145])
            
            if curtransFlag == 'Y' and self.augtype['rotate']:
                imgs, label, weight = augment_random_rotate(imgs, label, weight, angle=10, threshold=0.7)
                imgs, label, weight = central_crop(imgs, label, weight, self.crop_size)
            
            imgs = (imgs.astype(np.float32)) / 255.0

        # Formatting for PyTorch
        imgs = imgs[np.newaxis, ...]
        label = label[np.newaxis, ...]
        
        if self.phase == 'train':
            weight = weight[np.newaxis, ...]
            return torch.from_numpy(imgs).float(), torch.from_numpy(label).float(), \
                   torch.from_numpy(weight).float(), curNameID
        else:
            return torch.from_numpy(imgs).float(), torch.from_numpy(label).float(), \
                   torch.from_numpy(origin), torch.from_numpy(spacing), \
                   [curNameID], [curSplitID], torch.from_numpy(np.array(curnzhw)), \
                   torch.from_numpy(np.array(curShapeOrg))