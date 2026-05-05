import os
import numpy as np
from glob import glob
import SimpleITK as sitk
import argparse
import tqdm


def lumTrans_hu(img):
	"""
	:param img: CT image
	:return: Hounsfield Unit window clipped and normalized
	"""
	lungwin = np.array([-1000.,600.])
	newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
	newimg[newimg < 0] = 0
	newimg[newimg > 1] = 1
	newimg = (newimg*255).astype('uint8')
	return newimg

def load_itk_image(filename):
	itkimage = sitk.ReadImage(filename)
	numpyImage = sitk.GetArrayFromImage(itkimage)
	numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
	numpySpacing = np.array(list(reversed(itkimage.GetSpacing()))) 
	return numpyImage, numpyOrigin, numpySpacing

def save_itk(image, origin, spacing, filename):
	"""
	:param image: images to be saved
	:param origin: CT origin
	:param spacing: CT spacing
	:param filename: save name
	:return: None
	"""
	if type(origin) is not tuple:
		if type(origin) is list:
			origin = tuple(reversed(origin))
		else:
			origin = tuple(reversed(origin.tolist()))
	if type(spacing) is not tuple:
		if type(spacing) is list:
			spacing = tuple(reversed(spacing))
		else:
			spacing = tuple(reversed(spacing.tolist()))
	itkimage = sitk.GetImageFromArray(image, isVector=False)
	itkimage.SetSpacing(spacing)
	itkimage.SetOrigin(origin)
	sitk.WriteImage(itkimage, filename, True)


def get_3d_bbox(mask, margin=5):
    """
    Finds the bounding box of a 3D binary mask and adds a margin.
    
    Args:
        mask (numpy.ndarray): 3D binary array (Z, Y, X).
        margin (int): Number of pixels to add to each side.
        
    Returns:
        tuple: (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    # Find indices where mask is non-zero
    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        return None  # Handle empty mask case

    # Get min and max for each axis
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    # Add margin and clip to image boundaries
    z_min = max(0, z_min - margin)
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    
    z_max = min(mask.shape[0] - 1, z_max + margin)
    y_max = min(mask.shape[1] - 1, y_max + margin)
    x_max = min(mask.shape[2] - 1, x_max + margin)

    return z_min, z_max, y_min, y_max, x_min, x_max

# Example Usage:
# bbox = get_3d_bbox(my_mask, margin=10)
# if bbox:
#     z1, z2, y1, y2, x1, x2 = bbox
#     cropped_image = original_image[z1:z2+1, y1:y2+1, x1:x2+1]

def get_cubic_bbox(mask, margin=5):
	bbox = get_3d_bbox(mask, margin)
	if not bbox:
		return None

	z1, z2, y1, y2, x1, x2 = bbox

	# Calculate current widths
	y_width = y2 - y1
	x_width = x2 - x1
	z_width = z2 - z1

	# Find the largest side to make it square
	max_side = max(y_width, x_width)
	if max_side < 128:
		max_side = 128
	
	if z_width < 128:
		# check if image size is greather than 128
		z_center = (z1 + z2) // 2
		if min(mask.shape[0], z_center + (128 // 2)) == mask.shape[0]:
			z_center = mask.shape[0] - (128 // 2)
		z1 = max(0, z_center - 128 // 2)
		z2 = min(mask.shape[0], z1 + 128)

	# Center the square crop
	y_center = (y1 + y2) // 2
	if min(mask.shape[1], y_center + (128 // 2)) == mask.shape[1]:
		y_center = mask.shape[1] - (128 // 2)

	x_center = (x1 + x2) // 2
	if min(mask.shape[2], x_center + (128 // 2)) == mask.shape[2]:
		x_center = mask.shape[2] - (128 // 2)
	
	new_y1 = max(0, y_center - max_side // 2)
	new_y2 = min(mask.shape[1], new_y1 + max_side)

	new_x1 = max(0, x_center - max_side // 2)
	new_x2 = min(mask.shape[2], new_x1 + max_side)

	return z1, z2, new_y1, new_y2, new_x1, new_x2

def clean_images(images_path):

	lungmask_clean_path = images_path.replace('image', 'lungmask_clean')
	image_clean_path = images_path.replace('image', 'image_clean')
	label_clean_path = images_path.replace('image', 'label_clean')
	smallairway_clean_path = images_path.replace('image', 'smallairway_clean')
	print(f"Saving paths\n{image_clean_path}\n{label_clean_path}\n{lungmask_clean_path}\n{smallairway_clean_path}")

	image_names = glob(os.path.join(images_path, '*.nii*'))

	for image_name in tqdm.tqdm(image_names):

		# receives image name
		name = image_name.split('/')[-1].split('.nii')[0]
		lungmask_name = name + '_lungmask.nii.gz'

		lungmask, origin, spacing  = load_itk_image(os.path.join(images_path.replace('image', 'lungmask'), lungmask_name))
		if min(lungmask.shape) < 128:
			print("Image smaller than minimum cube!")
			continue
		bbox = get_cubic_bbox(lungmask)

		if bbox is None:
			print(f"Could not get bbox for image {image_name}, skipping...")
		
		else:
			z1, z2, y1, y2, x1, x2 = bbox

			# crop lungmask
			cropped_lungmask = lungmask[z1:z2+1, y1:y2+1, x1:x2+1]
			data_savepath = os.path.join(lungmask_clean_path, lungmask_name)
			save_itk(cropped_lungmask, origin, spacing, data_savepath)
			del lungmask # save space

			# load image and crop
			image, _, _ = load_itk_image(os.path.join(images_path, image_name.split('/')[-1]))
			cropped_image = image[z1:z2+1, y1:y2+1, x1:x2+1]
			# perform HU windowing and intensities preprocessing
			cropped_image[np.isnan(cropped_image)] = -2000
			cropped_image_hu = lumTrans_hu(cropped_image)
			data_savepath = os.path.join(image_clean_path, name + '_clean_hu.nii.gz')
			save_itk(cropped_image_hu, origin, spacing, data_savepath)
			del cropped_image, cropped_image_hu # save space

			# load label and crop
			label_name = name + '_label.nii.gz'
			label, _, _ = load_itk_image(os.path.join(images_path.replace('image', 'label'), label_name))
			cropped_label = label[z1:z2+1, y1:y2+1, x1:x2+1]
			data_savepath = os.path.join(label_clean_path, name + '_label.nii.gz')
			save_itk(cropped_label, origin, spacing, data_savepath)
			del cropped_label # save space

			# load smallairway and crop
			airway_name = name + '_smallairway.nii.gz'
			airway, _, _ = load_itk_image(os.path.join(images_path.replace('image', 'smallairway'), airway_name))
			cropped_airway = airway[z1:z2+1, y1:y2+1, x1:x2+1]
			data_savepath = os.path.join(smallairway_clean_path, name + '_smallairway.nii.gz')
			save_itk(cropped_airway, origin, spacing, data_savepath)
			del cropped_airway # save space


if __name__=='__main__':
	parser = argparse.ArgumentParser(
		prog='Preprocess images to crop region of interest for TfeNet',
		description='This code gets images from image, label and lungmask folders and create the clean version of it',
		epilog='Get started!'
	)
	parser.add_argument('-f', '--folder', type=str, required=True, help="Folder path of the datatset")

	args = parser.parse_args()

	print(args)
	assert os.path.exists(args.folder), "Source folder for images does not exist"

	folders_list = ['image', 'label', 'lungmask', 'smallairway']
	subfolders_list = ['train', 'test', 'val']	

	for f in folders_list:
		assert os.path.exists(os.path.join(args.folder, f)), f"'{f}' folder does not exist"
		for sub in subfolders_list:
			assert os.path.exists(os.path.join(args.folder, f, sub)), f"'{sub}' does not exist in '{f}' folder"

	# create folders for preprocessed data
	for f in folders_list:
		for sub in subfolders_list:
			os.makedirs(os.path.join(args.folder, f'{f}_clean', sub), exist_ok=True)

	for sub in subfolders_list:
		clean_images(os.path.join(args.folder, 'image', sub))
	print("Finished preprocessing images!")