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
    def __init__(self, config, phase='train', stage=1, split_comber=None,
                 debug=False, crop_size=[128, 128, 128], random_select=False, small_airway=False):
        
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.augtype = config['augtype']
        self.split_comber = split_comber
        self.patch_per_case = 16
        self.phase = phase
        self.crop_size = crop_size
        self.small_airway = small_airway
        self.datapath = config['dataset_path']
        
        # Determine paths
        sub_folder = 'train' if phase == 'train' else ('val' if phase == 'val' else 'test')
        self.base_path = os.path.join(self.datapath, 'image_clean', sub_folder)
        self.data_file_names = sorted(os.listdir(self.base_path))

        # Shuffle the filenames once at the very start
        if self.phase == 'train':
            random.shuffle(self.data_file_names)
        
        if debug:
            self.data_file_names = self.data_file_names[:1]
        
        self.caseNumber = len(self.data_file_names)

        # For Val/Test, we still need the cubelist for splitting 3D volumes
        self.cubelist = []
        if self.phase != 'train':
            print(f"--- Preparing {self.phase} metadata ---")
            for i, file_name in enumerate(self.data_file_names):
                raw_path = os.path.join(self.base_path, file_name)
                imgs, _, _ = load_itk_image(raw_path)
                splits, nzhw, orgshape = self.split_comber.split_id(imgs)
                data_name = file_name.split('.nii')[0]
                for j in range(len(splits)):
                    self.cubelist.append({'name': data_name, 'split': splits[j], 'id': j, 
                                         'nzhw': nzhw, 'org': orgshape, 'file_idx': i})
    
    def shuffle_dataset(self):
        """
        Call this at the beginning of your training loop epoch 
        to get a new order of images.
        """
        if self.phase == 'train':
            random.shuffle(self.data_file_names)

    def __len__(self):
        return self.patch_per_case * self.caseNumber if self.phase == 'train' else len(self.cubelist)
    
    @lru_cache(maxsize=1) # One image per worker to save RAM
    def _load_case(self, data_name, file_idx):
        """
        This is the most stable way to load. 
        It loads 1 case, keeps it in RAM for 16 patches, 
        then throws it away to make room for the next.
        """
        fname = self.data_file_names[file_idx]
        raw_path = os.path.join(self.base_path, fname)
        
        # Path logic (keep your existing logic here)
        if self.small_airway:
            label_path = raw_path.replace('image_clean', 'smallairway_clean').replace('clean_hu', 'smallairway')
            weight_path = raw_path.replace('image_clean', 'LIB_weight_small').replace('clean_hu.nii.gz', 'smallweight.npy')
        else:
            label_path = raw_path.replace('image', 'label').replace('clean_hu', 'label')
            weight_path = raw_path.replace('image_clean', 'LIB_weight').replace('clean_hu.nii.gz', 'weight.npy')

        imgs, origin, spacing = load_itk_image(raw_path)
        labels, _, _ = load_itk_image(label_path)
        weight = np.load(weight_path) if self.phase == 'train' else None
        
        # We store as float16 in RAM to save 50% memory
        return {
            'imgs': imgs.astype(np.float16), 
            'labels': labels.astype(np.uint8), 
            'weight': weight.astype(np.float16) if weight is not None else None,
            'origin': origin, 
            'spacing': spacing
        }

    def __getitem__(self, idx):
        if self.phase == 'train':
            file_idx = idx // self.patch_per_case
            data_name = self.data_file_names[file_idx].split('.nii')[0]
            
            # Fetch from cache
            data = self._load_case(data_name, file_idx)
            
            # IMMEDIATELY .copy() and convert back to float32 for processing
            imgs = data['imgs'].astype(np.float32).copy()
            label = (data['labels'] > 0).astype(np.float32).copy()
            weight = data['weight'].astype(np.float32).copy()

            # --- Robust Cropping to prevent "high <= 0" error ---
            # If image is smaller than target, we adjust sampling or pad
            target = [128, 128, 128]
            # sample_size = [max(target[i], imgs.shape[i]) for i in range(3)] 
            # Note: You might need to add padding here if images < 128
            
            if any(imgs.shape[i] <= 145 for i in range(3)):
                imgs, label, weight = random_sample(imgs, label, weight, target)
            else:
                imgs, label, weight = random_sample(imgs, label, weight, [145, 145, 145])

            if self.augtype['rotate']:
                imgs, label, weight = augment_random_rotate(imgs, label, weight, angle=10, threshold=0.7)
                imgs, label, weight = central_crop(imgs, label, weight, self.crop_size)

            imgs = (imgs.astype(np.float32)) / 255.0
            
            return torch.from_numpy(imgs[None]).float(), \
                   torch.from_numpy(label[None]).float(), \
                   torch.from_numpy(weight[None]).float(), data_name


            '''self._update_buffer(file_idx) # Ensure image is in RAM
            
            data = self.current_buffer[data_name]
            
            imgs, label, weight = data['imgs'].copy(), data['labels'].copy(), data['weight'].copy()
            label = (label > 0).astype('float')
            
            # --- Robust Cropping to prevent "high <= 0" error ---
            # If image is smaller than target, we adjust sampling or pad
            target = [128, 128, 128]
            # sample_size = [max(target[i], imgs.shape[i]) for i in range(3)] 
            # Note: You might need to add padding here if images < 128
            
            if any(imgs.shape[i] <= 145 for i in range(3)):
                imgs, label, weight = random_sample(imgs, label, weight, target)
            else:
                imgs, label, weight = random_sample(imgs, label, weight, [145, 145, 145])

            if self.augtype['rotate']:
                imgs, label, weight = augment_random_rotate(imgs, label, weight, angle=10, threshold=0.7)
                imgs, label, weight = central_crop(imgs, label, weight, self.crop_size)

            imgs = (imgs.astype(np.float32)) / 255.0

            imgs = imgs[np.newaxis, ...]
            label = label[np.newaxis, ...]
            weight = weight[np.newaxis, ...]

            return torch.from_numpy(imgs).float(), torch.from_numpy(label).float(), \
                   torch.from_numpy(weight).float(), data_name'''

        else:
            # Logic for Val/Test using cubelist
            item = self.cubelist[idx]
            curNameID = item['name']
            cursplit = item['split']     # Coordinates: [[z1, z2], [y1, y2], [x1, x2]]
            curSplitID = item['id']
            curnzhw = item['nzhw']
            curShapeOrg = item['org']
            file_idx = item['file_idx']  # We added this to know which file to load

            # Fetch the full volume from cache (or disk)
            # Using the same cached function keeps Val fast!
            data = self._load_case(curNameID, file_idx)
            
            # Slice the specific cube coordinates
            z, y, x = cursplit
            imgs = data['imgs'][z[0]:z[1], y[0]:y[1], x[0]:x[1]].astype(np.float32).copy()
            label = (data['labels'][z[0]:z[1], y[0]:y[1], x[0]:x[1]] > 0).astype(np.float32).copy()

            # 4. Normalization (No random sampling/augmentation for Val)
            imgs = imgs / 255.0
            
            return torch.from_numpy(imgs[None]).float(), \
               torch.from_numpy(label[None]).float(), \
               torch.from_numpy(data['origin']), \
               torch.from_numpy(data['spacing']), \
               [curNameID], [curSplitID], \
               torch.from_numpy(np.array(curnzhw)), \
               torch.from_numpy(np.array(curShapeOrg))
