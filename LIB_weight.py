# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 19:23:00 2021

@author: Hao Zheng
"""
import numpy as np
import os
from scipy import ndimage
from utils import load_itk_image
from tqdm import tqdm
import argparse

def neighbor_descriptor(label, filters):
    den = filters.sum()
    conv_label = ndimage.convolve(label.astype(np.float32), filters, mode='mirror')/den
    conv_label[conv_label==0] = 1
    conv_label = -np.log10(conv_label)
    return conv_label

def save_local_imbalance_based_weight(label_path, save_path, small_airway=False):
    file_list = os.listdir(label_path)
    file_list.sort()
    filter0 = np.ones([7,7,7], dtype=np.float32)
    for i in tqdm(range(len(file_list))):
        label,_,_ = load_itk_image(os.path.join(label_path, file_list[i])) #load the binary labels
        weight = neighbor_descriptor(label, filter0)  
        weight[weight>1]=1.     
        # weight = weight*label
        #Here is constant weight. During training, varied weighted training is adopted.
        #weight = weight**np.random.random(2,3) * label + (1-label) in dataloader.
        # weight = weight**2.5 
        weight = weight.astype(np.float32)
        if small_airway:
            save_name = os.path.join(save_path,file_list[i].split('_smallairway')[0] + "_smallweight.npy")    
        else:
            save_name = os.path.join(save_path,file_list[i].split('_label')[0] + "_weight.npy")
        np.save(save_name, weight) 
        print(file_list[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Imbalance weights for TfeNet',
        description='Preprocess imbalance of foreground and background on the dataset for TfeNet',
        epilog='Get started!'
    )
    parser.add_argument('-lcf', '--label_clean_folder', type=str, required=True, help="Folder path to label_clean/train inside dataset folder for TfeNet")
    parser.add_argument('-lwf', '--lib_weight_folder', type=str, required=True, help="Folder path to LIB_weight/train inside dataset folder for TfeNet")
    parser.add_argument('--small_airways', default=False, action='store_true', help='if set, will work on smallairway labels instead of labels')

    args = parser.parse_args()
    print(args)

    save_local_imbalance_based_weight(args.label_clean_folder, args.lib_weight_folder, args.small_airways)
  