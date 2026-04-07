from lungmask import LMInferer
import SimpleITK as sitk
import os
from utils import load_itk_image, save_itk
from tqdm import tqdm
import argparse


def ex_lungmask(src_path, save_path):
    img_files = os.listdir(src_path)

    for i in tqdm(range(len(img_files))):
        name = img_files[i].split('/')[-1].split('.nii')[0]
        path = os.path.join(src_path, img_files[i])     
        img, oring, spacing = load_itk_image(path)

        inferer = LMInferer()
        lungmask = inferer.apply(img)  # default model is U-net(R231)

        path = os.path.join(save_path, name + '_lungmask.nii.gz')     
        print(path)
        save_itk(lungmask, oring, spacing, path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Extract lungmask for TfeNet',
        description='This code extracts the lungmask of thei images in folder and saves it in destination',
        epilog='Get started!'
    )
    parser.add_argument('-f', '--folder', type=str, required=True, help="Folder path with 3D images in the nnUNet raw format")
    parser.add_argument('-d', '--destination', type=str, required=True, help="Folder to save images and masks in TfeNet format")

    args = parser.parse_args()

    print(args)

    assert os.path.exists(args.folder), "Source folder for images does not exist"
    assert os.path.exists(args.destination), "Destination folder for images does not exist"
    
    ex_lungmask(args.folder, args.destination)
    print(f"Finished inferencing lungmask for images in {args.folder}")
