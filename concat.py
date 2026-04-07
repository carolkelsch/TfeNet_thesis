import os 
from utils import load_itk_image, save_itk
from tqdm import tqdm
import numpy as np
import argparse


def concat_airway(label_path, small_label_path, save_path):
    label_files = os.listdir(label_path)

    for i in tqdm(range(len(label_files))):
        print(label_files[i])
        name = label_files[i].split('/')[-1].split('.nii')[0]
        path = os.path.join(label_path,label_files[i])     
        label , oring , spacing = load_itk_image(path)
        path = os.path.join(small_label_path,label_files[i])   
        small_label ,_ , _ = load_itk_image(path)
        concat = label + small_label  
        concat[concat>0]=1
        concat = concat.astype(np.uint8)
        path = os.path.join(save_path,name + '.nii.gz')     
        save_itk(concat,oring,spacing,path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Evaluate generated model on validation',
        description='This code predicts the segmeentation for the validation set with TfeNet',
        epilog='Get started!'
    )
    parser.add_argument('-pred', '--predictions_folder', type=str, required=True, help="Folder path to the predicted images")
    parser.add_argument('-pred_small', '--predictions_small_folder', type=str, required=True, help="Folder path to small airway predicted images")
    parser.add_argument('-s', '--saving_folder', type=str, required=True, help="Path to save the concatenated outputs")

    args = parser.parse_args()

    print(args)
    concat_airway(args.predictions_folder, args.predictions_small_folder, args.saving_folder)