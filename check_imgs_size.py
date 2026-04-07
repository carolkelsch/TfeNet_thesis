import SimpleITK as sitk
import os
import argparse


def check_image_dimensions(folder_path):
    # 1. Get a list of all files in the directory
    # We filter for common image extensions to avoid errors
    extensions = ('.nrrd', '.mha', '.nii', '.nii.gz', '.tif', '.png', '.jpg')
    files = [f for f in os.listdir(folder_path) if f.endswith(extensions)]

    if not files:
        print("No supported image files found in the directory.")
        return

    print(f"{'File Name':<30} | {'Size (Width, Height, Depth)':<30}")
    print("-" * 65)

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # 2. Read the image
            image = sitk.ReadImage(file_path)
            
            # 3. Get the size (SimpleITK uses Width, Height, Depth order)
            size = image.GetSize()
            
            print(f"{file_name:<30} | {str(size):<30}")
            
        except Exception as e:
            print(f"Could not read {file_name}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Check images sizes',
        description='This code opens images in a folder and prints the dimensions',
        epilog='Get started!'
    )
    parser.add_argument('-f', '--folder', type=str, required=True, help="Folder path")
    
    args = parser.parse_args()

    check_image_dimensions(args.folder)