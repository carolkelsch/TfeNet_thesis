import os 
from utils import load_itk_image, save_itk
from tqdm import tqdm
import argparse
import numpy as np

def ex_small_airway(label_path, save_path):

    os.makedirs(save_path, exist_ok=True)

    label_files = os.listdir(label_path)
    for i in tqdm(range(len(label_files))):
        name = label_files[i].split('/')[-1].split('_label')[0]
        path1 = os.path.join(label_path, label_files[i])
        print(path1)
        label, oring, spacing = load_itk_image(path1)
        path2 = path1.replace('label', 'lungmask')
        print(path2)
        mask ,_ , _ = load_itk_image(path2)
        binary_mask = (mask > 0).astype(int)
        sm = label * binary_mask
        path3 = os.path.join(save_path, name + '_smallairway.nii.gz')
        print(path3)
        save_itk(sm.astype(np.uint8), oring, spacing, path3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Filter airways by the lungmask for TfeNet',
        description='This code extracts the small airways',
        epilog='Get started!'
    )
    parser.add_argument('-f', '--folder', type=str, required=True, help="Folder path to label inside dataset folder for TfeNet")
    parser.add_argument('-d', '--destination', type=str, required=True, help="Folder path to smallairway inside dataset folder for TfeNet")

    args = parser.parse_args()

    folders_list = ['train', 'test', 'val']

    assert os.path.exists(args.folder), "Source folder for images does not exist"
    assert os.path.exists(args.destination), "Destination folder for images does not exist"
    
    folders = os.listdir(args.folder)

    for f in folders_list:
        assert os.path.exists(os.path.join(args.folder, f)), f"'{f}' folder does not exist"

    print(args)

    for f in folders_list:
        ex_small_airway(os.path.join(args.folder, f), os.path.join(args.destination, f))
    print("Finished generating smallairways!")