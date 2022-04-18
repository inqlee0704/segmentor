# Given predicted mask and ground truth, it calculates accuracy
import os
from tqdm.auto import tqdm
import wandb
import pandas as pd
import numpy as np
import argparse

# ML
import torch
from torch import nn

# Others
import SimpleITK as sitk
from medpy.io import load
sitk.ProcessObject_SetGlobalWarningDisplay(False)

parser = argparse.ArgumentParser(description='segmentor')
parser.add_argument('--mask', default='lobes', type=str, help='[airway, vessels, lung, lobes]')
parser.add_argument('--mask_name', default='ZUNet_lobes_sigma3.img.gz', type=str, help='mask file name')
parser.add_argument('--label_name', default='ZUNU_lobes.img.gz', type=str, help='label file name')
parser.add_argument('--subj_path', default='', type=str, help='Subject path, ex) VIDA_*/24')
parser.add_argument('--in_file_path',
    default='D:/silicosis/data/TE_ProjSubjListDCM.in',
    type=str,
    help='path to *.in')

def get_config(args):
    config = wandb.config
    config.in_file_path = args.in_file_path
    config.subj_path = args.subj_path
    config.mask = args.mask # 'airway', 'lung', 'lobes'
    config.mask_name = args.mask_name
    config.label_name = args.label_name

    if args.mask == 'lobes':
        config.num_c = 6
    elif args.mask == 'lung':
        config.num_c = 3
    else:
        config.num_c = 2
    
    return config


def get_stats(one_hot_pred, one_hot_targets, dim):
    tp = (one_hot_pred * one_hot_targets).sum()
    fp = one_hot_pred.sum() - tp
    fn = one_hot_targets.sum() - tp
    tn = torch.prod(torch.tensor(dim)) - (tp + fp + fn)
    return tp, fp, fn, tn

def get_sensitivity(tp,fn): # Recall
    return tp.float()/(tp.float()+fn.float()+1e-8)

def get_specificity(tn,fp):
    return tn.float()/(tn.float()+fp.float()+1e-8)

def get_precision(tp,fp):
    return tp.float()/(tp.float()+fp.float()+1e-8)

def get_dice_score(precision, sensitivity):
    return 2 * (precision * sensitivity)/(precision + sensitivity)

def get_lobe_stats(pred,label):
    one_hot_targets = nn.functional.one_hot(torch.from_numpy(label).to(torch.int64),num_classes=6)
    one_hot_pred = nn.functional.one_hot(torch.from_numpy(pred).to(torch.int64),num_classes=6)
    
    dices = []
    spes = []
    sens = []
    for i in range(6):
        lobe_pred = one_hot_pred[:,:,:,i]
        lobe_target =  one_hot_targets[:,:,:,i]
        tp,fp,fn,tn = get_stats(lobe_pred, lobe_target, lobe_pred.shape)
        sensitivity = get_sensitivity(tp,fn)
        specificity = get_specificity(tn,fp)
        precision = get_precision(tp,fp)
        dice = get_dice_score(precision,sensitivity)
        sens.append(sensitivity)
        spes.append(specificity)
        dices.append(dice)
    return dices, sens, spes


def main(config):
    infer_df = pd.read_csv(config.in_file_path, sep='\t')
    dice1 = []
    dice2 = []
    dice3 = []
    dice4 = []
    dice5 = []
    sensitivity1 = []
    sensitivity2 = []
    sensitivity3 = []
    sensitivity4 = []
    sensitivity5 = []
    specificity1 = []
    specificity2 = []
    specificity3 = []
    specificity4 = []
    specificity5 = []
    pbar = tqdm(range(len(infer_df)))
    for i in pbar:
        subj_path = infer_df.ImgDir[i]
        pbar.set_description(subj_path)
        mask_path = os.path.join(subj_path,config.mask_name)
        label_path = os.path.join(subj_path,config.label_name)

        if not os.path.exists(mask_path):
            print(f"{mask_path} does not exist.")
            return
        if not os.path.exists(label_path):
            print(f"{label_path} does not exist.")
            return 
        
        mask, _ = load(mask_path)
        label, _ = load(label_path)

        if config.mask == 'lobes':
            mask[mask==8] = 1
            mask[mask==16] = 2
            mask[mask==32] = 3
            mask[mask==64] = 4
            mask[mask==128] = 5

            label[label==8] = 1
            label[label==16] = 2
            label[label==32] = 3
            label[label==64] = 4
            label[label==128] = 5

        dice, sensitivity, specificity = get_lobe_stats(mask.astype(np.uint8),label.astype(np.uint8))
        dice1.append(dice[1])
        dice2.append(dice[2])
        dice3.append(dice[3])
        dice4.append(dice[4])
        dice5.append(dice[5])

        sensitivity1.append(sensitivity[1])
        sensitivity2.append(sensitivity[2])
        sensitivity3.append(sensitivity[3])
        sensitivity4.append(sensitivity[4])
        sensitivity5.append(sensitivity[5])

        specificity1.append(specificity[1])
        specificity2.append(specificity[2])
        specificity3.append(specificity[3])
        specificity4.append(specificity[4])
        specificity5.append(specificity[5])

    print('-----')
    print('Dice:')
    print('-----')
    print(f'LUL: {np.mean(dice1):.3f} \u00B1 {np.std(dice1):.3f}')
    print(f'LLL: {np.mean(dice2):.3f} \u00B1 {np.std(dice2):.3f}')
    print(f'RUL: {np.mean(dice3):.3f} \u00B1 {np.std(dice3):.3f}')
    print(f'RML: {np.mean(dice4):.3f} \u00B1 {np.std(dice4):.3f}')
    print(f'RLL: {np.mean(dice5):.3f} \u00B1 {np.std(dice5):.3f}')

    print('------------')
    print('Sensitivity:')
    print('------------')
    print(f'LUL: {np.mean(sensitivity1):.3f} \u00B1 {np.std(sensitivity1):.3f}')
    print(f'LLL: {np.mean(sensitivity2):.3f} \u00B1 {np.std(sensitivity2):.3f}')
    print(f'RUL: {np.mean(sensitivity3):.3f} \u00B1 {np.std(sensitivity3):.3f}')
    print(f'RML: {np.mean(sensitivity4):.3f} \u00B1 {np.std(sensitivity4):.3f}')
    print(f'RLL: {np.mean(sensitivity5):.3f} \u00B1 {np.std(sensitivity5):.3f}')

    print('------------')
    print('Specificity:')
    print('------------')
    print(f'LUL: {np.mean(specificity1):.3f} \u00B1 {np.std(specificity1):.3f}')
    print(f'LLL: {np.mean(specificity2):.3f} \u00B1 {np.std(specificity2):.3f}')
    print(f'RUL: {np.mean(specificity3):.3f} \u00B1 {np.std(specificity3):.3f}')
    print(f'RML: {np.mean(specificity4):.3f} \u00B1 {np.std(specificity4):.3f}')
    print(f'RLL: {np.mean(specificity5):.3f} \u00B1 {np.std(specificity5):.3f}')

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args)
    main(config)