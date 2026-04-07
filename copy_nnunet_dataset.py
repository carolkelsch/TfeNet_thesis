import SimpleITK as sitk
import numpy as np
import argparse
import os
from tqdm import tqdm
import shutil

def get_standardized_volume(file_path):
    image = sitk.ReadImage(file_path)
    standard_img = sitk.DICOMOrient(image, 'RPS')
    
    return standard_img

def resample_data(image_file, new_spacing=[1.0, 1.0, 1.0]):
    
    image = get_standardized_volume(image_file)
    
    original_spacing = image.GetSpacing()
    if original_spacing != new_spacing:
        # resample image to normalized dimensions
        original_size = image.GetSize()

        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        
        image = resampler.Execute(image)
    
    return image

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Preprocess dataset for TfeNet',
        description='This code generate the splits and resamples the images to 1mmx1mmx1mm for lungmask and subsequent TfeNet processings',
        epilog='Get started!'
    )
    parser.add_argument('-f', '--folder', type=str, required=True, help="Folder path with 3D images in the nnUNet raw format")
    parser.add_argument('-d', '--destination', type=str, required=True, help="Folder to save images and masks in TfeNet format")

    args = parser.parse_args()

    print(args)
    np.random.seed(42)  # For reproducibility

    # get subfolders
    sub_folders = [f for f in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, f))]
    # check expected folder exist
    assert 'imagesTr' in sub_folders, "imagesTr was not found in the provided folder"
    assert 'imagesTv' in sub_folders, "imagesTv was not found in the provided folder"
    assert 'imagesTs' in sub_folders, "imagesTs was not found in the provided folder"
    assert 'labelsTr' in sub_folders, "labelsTr was not found in the provided folder"
    assert 'labelsTv' in sub_folders, "labelsTv was not found in the provided folder"
    assert 'labelsTs' in sub_folders, "labelsTs was not found in the provided folder"

    # destination folders in the format expected by TfeNet
    '''
    ./BAS/
    ├── image
    │   ├── test
    │   ├── train
    │   └── val
    ├── image_clean
    │   ├── test
    │   ├── train
    │   └── val
    ├── label
    │   ├── test
    │   ├── train
    │   └── val
    ├── label_clean
    │   ├── test
    │   ├── train
    │   └── val
    ├── LIB_weight
    │   └── train
    ├── LIB_weight_small
    │   └── train
    ├── lungmask
    │   ├── test
    │   ├── train
    │   └── val
    ├── lungmask_clean
    │   ├── test
    │   ├── train
    │   └── val
    ├── smallairway
    │   └── train
    └── smallairway_clean
        ├── test
        ├── train
        └── val
    '''
    os.makedirs(os.path.join(args.destination, 'image', 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.destination, 'image', 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.destination, 'image', 'val'), exist_ok=True)

    os.makedirs(os.path.join(args.destination, 'label', 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.destination, 'label', 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.destination, 'label', 'val'), exist_ok=True)
    
    for folder in tqdm(sub_folders):
        if folder == 'imagesTr':
            dest_path = os.path.join(args.destination, 'image', 'train')
        elif folder == 'imagesTs':
            dest_path = os.path.join(args.destination, 'image', 'test')
        elif folder == 'imagesTv':
            dest_path = os.path.join(args.destination, 'image', 'val')
        elif folder == 'labelsTr':
            dest_path = os.path.join(args.destination, 'label', 'train')
        elif folder == 'labelsTs':
            dest_path = os.path.join(args.destination, 'label', 'test')
        elif folder == 'labelsTv':
            dest_path = os.path.join(args.destination, 'label', 'val')
        else:
            print("Found unusual folder, halting execution!")
            break
        
        # will get images and resample to 1mmx1mmx1mm spacing
        images_path = os.path.join(args.folder, folder)

        images_list = os.listdir(images_path)

        for image_name in tqdm(images_list):
            src_path = os.path.join(images_path, image_name)
            if folder.startswith('labels'):
                image_name = image_name[:-7] + "_0000_label" + image_name[-7:]
            dst_path = os.path.join(dest_path, image_name)
            shutil.copy(src_path, dst_path)
    
    print("Finished copying images!")