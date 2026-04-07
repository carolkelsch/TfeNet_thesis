import csv
import os
import numpy as np
import skimage.measure as measure
from skimage.morphology import skeletonize_3d
import SimpleITK as sitk
from scipy import ndimage
from utils import load_itk_image, save_itk
import argparse

EPSILON = 1e-32

def large_connected_domain(label):
    cd, num = measure.label(label, return_num = True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd==(k+1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    #print(volume_sort)
    label = (cd==(volume_sort[-1]+1)).astype(np.uint8)
    label = ndimage.binary_fill_holes(label)
    label = label.astype(np.uint8)
    return label
 
def skeleton_parsing(skeleton):
    # separate the skeleton
    neighbor_filter = ndimage.generate_binary_structure(3, 3)
    skeleton_filtered = ndimage.convolve(skeleton, neighbor_filter) * skeleton
    # distribution = skeleton_filtered[skeleton_filtered>0]
    # plt.hist(distribution)
    skeleton_parse = skeleton.copy()
    skeleton_parse[skeleton_filtered>3] = 0
    con_filter = ndimage.generate_binary_structure(3, 3)
    cd, num = ndimage.label(skeleton_parse, structure = con_filter)
    #remove small branches
    for i in range(num):
        a = cd[cd==(i+1)]
        if a.shape[0]<5:
            skeleton_parse[cd==(i+1)] = 0
    cd, num = ndimage.label(skeleton_parse, structure = con_filter)
    return skeleton_parse, cd, num

def tree_parsing_func(skeleton_parse, label, cd):
    #parse the airway tree
    edt, inds = ndimage.distance_transform_edt(1-skeleton_parse, return_indices=True) # 距离变换
    tree_parsing = np.zeros(label.shape, dtype = np.uint16)
    tree_parsing = cd[inds[0,...], inds[1,...], inds[2,...]] * label
    return tree_parsing

def get_parsing(mask):
    mask = (mask > 0).astype(np.uint8)
    mask = large_connected_domain(mask)
    skeleton = skeletonize_3d(mask)
    skeleton_parse, cd, num = skeleton_parsing(skeleton)
    tree_parsing = tree_parsing_func(skeleton_parse, mask, cd)
    return tree_parsing

def compute_binary_iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred) + EPSILON
    union = np.sum(y_true) + np.sum(y_pred) - intersection + EPSILON
    iou = intersection / union
    return iou

def evaluation_metrics(name, label, pred, postprocess=False):
    """
    Input: name用于处理显示当前处理的样本,label和 pred均为narray数据
    Output: 交并比IOU,气道检测长度比DLR,气道检测分支率DBR,相似系数DSC,精度precision,敏感度sensitivity
            特异性specificity,泄露率(假阳性率)leakages
    """
    if len(pred.shape) > 3:
        pred = pred[0]
    if len(label.shape) > 3:
        label = label[0]
    
    # compute tree sparsing
    parsing_gt = get_parsing(label)

    if postprocess:
        # find the largest component to locate the airway prediction
        cd, num = measure.label(pred, return_num=True, connectivity=2)
        volume = np.zeros([num])
        for k in range(num):
            volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
        volume_sort = np.argsort(volume)
        large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8) # large_cd 获取预测图像的最大连通区域

        large_cd = ndimage.binary_fill_holes(large_cd)
        large_cd = large_cd.astype(np.uint8)
        # large_cd = pred # test 无需后处理

        iou = compute_binary_iou(label, large_cd)

        # images were already postprocessed
        flag=-1
        while iou < 0.1:
            print(name," failed cases, require post-processing")
            large_cd = (cd == (volume_sort[flag-1] + 1)).astype(np.uint8)
            flag -= 1
            iou = compute_binary_iou(label, large_cd)

    else:
        large_cd = pred
        iou = compute_binary_iou(label, large_cd)

    skeleton = skeletonize_3d(label)
    skeleton = (skeleton > 0)
    skeleton = skeleton.astype(np.uint8)
    
    DLR = (large_cd * skeleton).sum() / skeleton.sum()

    precision = (large_cd * label).sum() / large_cd.sum()
    leakages = ((large_cd - label)==1).sum() / label.sum()
    sensitivity = (large_cd * label).sum() / label.sum()
    specificity = ((1 - large_cd) * (1 - label)).sum() /  (1 - label).sum()

    num_branch = parsing_gt.max()
    detected_num = 0
    for j in range(num_branch):
        branch_label = ((parsing_gt == (j + 1)).astype(np.uint8)) * skeleton # 取出每一个气道分支，并获取其分支骨架
        if (large_cd * branch_label).sum() / branch_label.sum() >= 0.8: # 若预测的长度大于等80%，说明该分支被检测到了
            detected_num += 1
    DBR = detected_num / num_branch

    DSC = 2 * (large_cd * label).sum() / ((large_cd + label).sum() + 1)

    print('name = %s, iou = %.4f, DLR = %.4f, DBR = %.4f, DSC = %.4f, precision = %.4f, sensitivity = %.4f, specificity = %.4f, leakages = %.4f'
          %(name, iou, DLR, DBR, DSC, precision, sensitivity, specificity, leakages))

    return iou, DLR, DBR, DSC, precision, sensitivity, specificity, leakages, large_cd

def evaluation(gt_clean_path, pred_path, save_path):
    file_names = os.listdir(gt_clean_path) # where images have ending _label.nii.gz
    name_list, iou_list, DLR_list, DBR_list, DSC_list, precision_list, sensitivity_list, specificity_list, leakages_list = [],[],[],[],[],[],[],[],[]
    label_names = [file for file in file_names if '_label' in file]

    for name in label_names:
        case_name = name.split('_label.nii')[0]
        # predictions have the ending _clean_hu.nii.gz
        label_path = os.path.join(gt_clean_path, name)
        pred_img_path = os.path.join(pred_path, case_name + "_clean_hu.nii.gz")
        
        pred, Origin, Spacing = load_itk_image(pred_img_path)
        label, _, _ = load_itk_image(label_path)
        
        iou, DLR, DBR, DSC, precision, sensitivity, specificity, leakages, large_cd = evaluation_metrics(case_name, label,  pred)
        
        name_list.append(case_name)
        iou_list.append(iou)
        DLR_list.append(DLR)
        DBR_list.append(DBR)
        DSC_list.append(DSC)
        precision_list.append(precision)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        leakages_list.append(leakages)
        save_name = os.path.join(save_path, case_name + '_post.nii.gz')
        save_itk(large_cd, Origin, Spacing, filename=save_name)

    # 平均值和标准差
    iou_mean = np.mean(iou_list)
    iou_std = np.std(iou_list)
    DLR_mean = np.mean(DLR_list)
    DLR_std = np.std(DLR_list)
    DBR_mean = np.mean(DBR_list)
    DBR_std = np.std(DBR_list)
    DSC_mean = np.mean(DSC_list)
    DSC_std = np.std(DSC_list)

    precision_mean = np.mean(precision_list)
    precision_std = np.std(precision_list)
    sensitivity_mean = np.mean(sensitivity_list)
    sensitivity_std = np.std(sensitivity_list)
    specificity_mean = np.mean(specificity_list)
    specificity_std = np.std(specificity_list)
    leakages_mean = np.mean(leakages_list)
    leakages_std = np.std(leakages_list)

    # 将结果保存为CSV文件
    with open(os.path.join(save_path, 'metric_results.csv'), 'a') as csvout:
        writer = csv.writer(csvout)
        row = ['name', 'iou', 'DLR', 'DBR', 'DSC', 'precision', 'sensitivity', 'specificity', 'leakages']
        writer.writerow(row)
        for i in range(len(iou_list)):
            row = [name_list[i], iou_list[i], DLR_list[i], DBR_list[i], DSC_list[i], precision_list[i],
                    sensitivity_list[i], specificity_list[i], leakages_list[i]]
            writer.writerow(row)
        row_mean = ['mean', iou_mean, DLR_mean, DBR_mean, DSC_mean, precision_mean, sensitivity_mean,
                    specificity_mean, leakages_mean]
        writer.writerow(row_mean)
        row_std = ['std', iou_std, DLR_std, DBR_std, DSC_std, precision_std, sensitivity_std, specificity_std,
                    leakages_std]
        writer.writerow(row_std)
        csvout.close()

def my_evaluation_metrics(name, label, pred):
    """
    Input: name用于处理显示当前处理的样本,label和 pred均为narray数据
    Output: 交并比IOU,气道检测长度比DLR,气道检测分支率DBR,相似系数DSC,精度precision,敏感度sensitivity
            特异性specificity,泄露率(假阳性率)leakages
    """
    if len(pred.shape) > 3:
        pred = pred[0]
    if len(label.shape) > 3:
        label = label[0]
    
    iou = compute_binary_iou(label, pred)

    tp = (pred * label).sum()
    tn = ((1 - pred) * (1 - label)).sum()
    fp = ((pred - label)==1).sum()
    fn = ((label - pred)==1).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    leakages = fp / (fp + tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    DSC = 2 * tp / (2 * tp + fp + fn)

    print('name = %s, iou = %.4f, DSC = %.4f, accuracy = %.4f, precision = %.4f, sensitivity = %.4f, specificity = %.4f, leakages = %.4f'
          %(name, iou, DSC, accuracy, precision, sensitivity, specificity, leakages))

    return iou, DSC, accuracy, precision, sensitivity, specificity, leakages

def my_evaluation(gt_clean_path, pred_path, save_path):
    file_names = os.listdir(gt_clean_path) # where images have ending _label.nii.gz
    name_list, iou_list, DSC_list, accuracy_list, precision_list, sensitivity_list, specificity_list, leakages_list = [],[],[],[],[],[],[],[]
    label_names = [file for file in file_names if '_label' in file]

    for name in label_names:
        case_name = name.split('_label.nii')[0]
        # predictions have the ending _clean_hu.nii.gz
        label_path = os.path.join(gt_clean_path, name)
        pred_img_path = os.path.join(pred_path, case_name + "_clean_hu.nii.gz")
        
        pred, Origin, Spacing = load_itk_image(pred_img_path)
        label, _, _ = load_itk_image(label_path)
        
        iou, DSC, accuracy, precision, sensitivity, specificity, leakages = my_evaluation_metrics(case_name, label,  pred)
        
        name_list.append(case_name)
        iou_list.append(iou)
        DSC_list.append(DSC)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        leakages_list.append(leakages)

    # 平均值和标准差
    iou_mean = np.mean(iou_list)
    iou_std = np.std(iou_list)
    DSC_mean = np.mean(DSC_list)
    DSC_std = np.std(DSC_list)

    accuracy_mean = np.mean(accuracy_list)
    accuracy_std = np.std(accuracy_list)
    precision_mean = np.mean(precision_list)
    precision_std = np.std(precision_list)
    sensitivity_mean = np.mean(sensitivity_list)
    sensitivity_std = np.std(sensitivity_list)
    specificity_mean = np.mean(specificity_list)
    specificity_std = np.std(specificity_list)
    leakages_mean = np.mean(leakages_list)
    leakages_std = np.std(leakages_list)

    # 将结果保存为CSV文件
    with open(os.path.join(save_path, 'tfenet_results.csv'), 'w') as csvout:
        writer = csv.writer(csvout)
        row = ['name', 'iou', 'DSC', 'accuracy', 'precision', 'sensitivity', 'specificity', 'leakages']
        writer.writerow(row)
        for i in range(len(iou_list)):
            row = [name_list[i], iou_list[i], DSC_list[i], accuracy_list[i], precision_list[i],
                    sensitivity_list[i], specificity_list[i], leakages_list[i]]
            writer.writerow(row)
        row_mean = ['mean', iou_mean, DSC_mean, accuracy_mean, precision_mean, sensitivity_mean,
                    specificity_mean, leakages_mean]
        writer.writerow(row_mean)
        row_std = ['std', iou_std, DSC_std, accuracy_std, precision_std, sensitivity_std, specificity_std,
                    leakages_std]
        writer.writerow(row_std)
        csvout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Evaluate generated model on validation',
        description='This code predicts the segmeentation for the validation set with TfeNet',
        epilog='Get started!'
    )
    parser.add_argument('-gt', '--gt_path', type=str, required=True, help="Folder path to the ground truth labels")
    parser.add_argument('-pred_outputs', '--pred_outputs_folder', type=str, required=True, help="Folder path to the predicted postprocessed images")
    parser.add_argument('-s', '--saving_folder', type=str, required=True, help="Path to save the outputs")

    args = parser.parse_args()

    print(args)
    my_evaluation(args.gt_path, args.pred_outputs_folder, args.saving_folder)