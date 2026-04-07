import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import random
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

		print("-------------------------Load all data into memory---------------------------")
		"""
		count the number of cases
		"""
		labellist = []
		cubelist = []
		self.caseNumber = 0
		allimgdata_memory = {}
		alllabeldata_memory = {}
		allweightdata_memory = {}

		if self.phase == 'train':
			# train_path = os.path.join(self.datapath + '\\image\\train')	
			train_path = os.path.join(self.datapath + '/image_clean/train')
			data_file_names = os.listdir(train_path)
			file_num = len(data_file_names)
			if self.debug_flag:
				data_file_names = data_file_names[:1]
				file_num = len(data_file_names)
			self.caseNumber += file_num
			self.data_file_names = data_file_names

			print("total %s case number: %d"%(self.phase, self.caseNumber))

			for file_name in data_file_names:
				raw_path = os.path.join(train_path, file_name)
				data_name = raw_path.split('/')[-1].split('.nii')[0]

				imgs, origin, spacing = load_itk_image(raw_path)

				if small_airway:
					label_path = raw_path.replace('image_clean', 'smallairway_clean')
					label_path = label_path.replace('clean_hu', 'smallairway')

					weight_path=raw_path.replace('image_clean', 'LIB_weight_small')
					weight_path=weight_path.replace('clean_hu.nii.gz', 'smallweight.npy')
				else:
					label_path = raw_path.replace('image', 'label')
					label_path = label_path.replace('clean_hu', 'label')

					weight_path=raw_path.replace('image_clean', 'LIB_weight')
					weight_path=weight_path.replace('clean_hu.nii.gz', 'weight.npy')
				
				labels, _, _ = load_itk_image(label_path)				
				weight = np.load(weight_path)

				allimgdata_memory[data_name] = [imgs, origin, spacing]
				alllabeldata_memory[data_name] = labels
				allweightdata_memory[data_name] = weight
				
				print("Name: %s "%(data_name),imgs.shape)
		

		elif self.phase == 'val':
			
			val_path = os.path.join(self.datapath + '/image_clean/val')
			data_file_names = os.listdir(val_path)
			file_num = len(data_file_names)
			if self.debug_flag:
				data_file_names = data_file_names[:1]
				file_num = len(data_file_names)
			self.caseNumber += file_num

			print("total %s case number: %d"%(self.phase, self.caseNumber))

			for file_name in data_file_names:
				raw_path = os.path.join(val_path, file_name)
				imgs, origin, spacing = load_itk_image(raw_path)
				splits, nzhw, orgshape = self.split_comber.split_id(imgs)
				data_name = raw_path.split('/')[-1].split('.nii')[0]
				
				print("Name: %s, # of splits: %d"%(data_name, len(splits)))

				if small_airway:
					label_path = raw_path.replace('image_clean', 'smallairway_clean')
					label_path = label_path.replace('clean_hu', 'smallairway')
				else:
					label_path = raw_path.replace('image', 'label')
					label_path = label_path.replace('clean_hu', 'label')
				
				labels, _, _ = load_itk_image(label_path)	
				
				allimgdata_memory[data_name] = [imgs, origin, spacing]
				alllabeldata_memory[data_name] = labels

				for j in range(len(splits)):
					cursplit = splits[j]
					curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
					cubelist.append(curlist)

		else:
			
			test_path = os.path.join(self.datapath + '/image_clean/test')
			data_file_names = os.listdir(test_path)
			file_num = len(data_file_names)
			if self.debug_flag:
				data_file_names = data_file_names[:1]
				file_num = len(data_file_names)
			self.caseNumber += file_num
			print("total %s case number: %d"%(self.phase, self.caseNumber))

			for file_name in data_file_names:
				raw_path = os.path.join(test_path, file_name)
				imgs, origin, spacing = load_itk_image(raw_path)
				# imgs = lumTrans(imgs) # 设置窗宽窗位为[-1000,600]
				splits, nzhw, orgshape = self.split_comber.split_id(imgs)
				data_name = raw_path.split('/')[-1].split('.nii')[0]
				print("Name: %s, # of splits: %d"%(data_name, len(splits)))

				if small_airway:
					label_path = raw_path.replace('image_clean', 'smallairway_clean')
					label_path = label_path.replace('clean_hu', 'smallairway')
				else:
					label_path = raw_path.replace('image', 'label')
					label_path = label_path.replace('clean_hu', 'label')

				labels, _, _ = load_itk_image(label_path)	

				allimgdata_memory[data_name] = [imgs, origin, spacing]
				alllabeldata_memory[data_name] = labels
				
				for j in range(len(splits)):
					"""
					check if this cube is suitable
					"""
					cursplit = splits[j]
					curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
					cubelist.append(curlist)

		self.allimgdata_memory = allimgdata_memory
		self.alllabeldata_memory = alllabeldata_memory
		self.allweightdata_memory = allweightdata_memory
		
		del allimgdata_memory,alllabeldata_memory,allweightdata_memory
		if self.phase == 'train':
			random.shuffle(data_file_names)
		elif self.phase == 'val':
			random.shuffle(cubelist)
			self.cubelist = cubelist
		elif self.phase == 'test':
			self.cubelist = cubelist
		print('---------------------Initialization Done---------------------')
		# print('Phase: %s total cubelist number: %d'%(self.phase, len(self.cubelist)))
		print()

	def __len__(self):
		"""
		:return: length of the dataset
		"""
		if self.phase == 'train':
			return self.patch_per_case*self.caseNumber
		else:
			return len(self.cubelist)

	def __getitem__(self, idx):
		"""
		:param idx: index of the batch
		:return: wrapped data tensor and name, shape, origin, etc.
		"""
		t = time.time()
		np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
		if self.phase != 'train':
			curlist = self.cubelist[idx]
			curNameID = curlist[0]
			cursplit = curlist[1]
			curSplitID = curlist[2]
			curnzhw = curlist[3]
			curShapeOrg = curlist[4]
			curtransFlag = curlist[5]
		else:
			data_name=self.data_file_names[idx//self.patch_per_case]
			curNameID = data_name.split('.nii')[0]
			curtransFlag = 'Y'

		label = self.alllabeldata_memory[curNameID]
		label = (label > 0)
		label = label.astype('float')
			
		if self.phase != 'train':
			label = label[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]	
			# small_labels = small_labels[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]	
		####################################################################
		
		if self.phase != 'train':
			imginfo = self.allimgdata_memory[curNameID]
			imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
			imgs = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
			imgs = (imgs.astype(np.float32))/255.0 
		else :
			weight = self.allweightdata_memory[curNameID]
			imginfo = self.allimgdata_memory[curNameID]
			imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]

			if imgs.shape[0] <= 145 or imgs.shape[1] <= 145 or imgs.shape[2] <= 145:
				imgs, label, weight = random_sample(imgs, label, weight, [128,128,128])
			else:
				imgs, label, weight = random_sample(imgs, label, weight, [145,145,145])
			
			if  curtransFlag == 'Y' and self.augtype['rotate'] is True:
				imgs, label, weight = augment_random_rotate(imgs, label, weight, angle=10,threshold=0.7)
				imgs, label, weight = central_crop(imgs, label, weight, self.crop_size)
			
			imgs = (imgs.astype(np.float32))/255.0 # 数据增强后进行归一化
		
		####################################################################

		imgs = imgs[np.newaxis,...]
		label = label[np.newaxis,...]
		
		if self.phase == 'train':
			weight = weight[np.newaxis,...]
			return torch.from_numpy(imgs).float(),torch.from_numpy(label).float(),\
				torch.from_numpy(weight).float(),curNameID
		else:
			curNameID = [curNameID]
			curSplitID = [curSplitID]
			curnzhw = np.array(curnzhw)
			curShapeOrg = np.array(curShapeOrg)
			return torch.from_numpy(imgs).float(),torch.from_numpy(label).float(),\
				torch.from_numpy(origin),\
				torch.from_numpy(spacing), curNameID, curSplitID,\
				torch.from_numpy(curnzhw),torch.from_numpy(curShapeOrg)