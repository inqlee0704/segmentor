import os
import sys
from dotenv import load_dotenv
from tqdm.auto import tqdm
import wandb
import numpy as np
import argparse

# Custom
from networks.UNet import UNet
from networks.ZUNet_v1 import ZUNet_v1
from engine import *
from dataloader import *
from losses import *

# ML
from torch.cuda import amp
import torch

# Others
import SimpleITK as sitk
from medpy.io import load, save
sitk.ProcessObject_SetGlobalWarningDisplay(False)

parser = argparse.ArgumentParser(description='segmentor')
parser.add_argument('--mask', default='lobe', type=str, help='[airway, vessels, lung, lobe]')
parser.add_argument('--model', default='ZUNet', type=str, help='[UNet, ZUNet]')
parser.add_argument('--in_file_path',
    default='D:/silicosis/data/TE_ProjSubjList.in',
    type=str,
    help='path to *.in')
parser.add_argument('--parameter_path',
    default='D:\segmentor\RESULTS\lobe\ZUNet_zerospadding_n294_20220124\ZUNet_zerospadding_n294_7.pth',
    type=str,
    help='path to *.pth')

args = parser.parse_args()

def get_config():
    config = wandb.config
    # ENV
    config.data_path = os.getenv("VIDA_PATH")
    config.in_file_path = args.in_file_path
    config.parameter_path = args.parameter_path
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.mask = args.mask # 'airway', 'lung', 'lobe'
    config.model = args.model
    config.Z = True
    config.num_c = 6
    config.in_c = 1
    
    return config


if __name__ == "__main__":
    load_dotenv()
    config = get_config()
    
    # load model
    if config.Z:
        # parameter_path = 'D:/segmentor/RESULTS/lobe/ZUNet_zerospadding_n294_20220124/ZUNet_zerospadding_n294_8.pth'
        parameter_path = config.parameter_path
        model = ZUNet_v1(in_channels=config.in_c, num_c=config.num_c)
        model.load_state_dict(torch.load(parameter_path))
        model.to(config.device)
        eng = Segmentor_Z(model=model,device=config.device)
    else:
        # parameter_path = 'D:/segmentor/RESULTS/UNet_reflectpadding_n32_20220120/UNet_reflectpadding_n32_42.pth'
        parameter_path = config.parameter_path

        model = UNet(in_channels=config.in_c, num_c=config.num_c)
        model.load_state_dict(torch.load(parameter_path))
        model.to(config.device)
        eng = Segmentor(model=model, device=config.device)
    
    infer_df = pd.read_csv(config.in_file_path, sep='\t')
    pbar = tqdm(range(len(infer_df)))
    for i in pbar:
        subj_path = infer_df.ImgDir[i]
        print(subj_path)
        img_path = os.path.join(subj_path,'zunu_vida-ct.img')
        img, hdr = load(img_path)
        img[img<-1024] = -1024
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        pred = eng.inference(img)
        pred[pred==1] = 8
        pred[pred==2] = 16
        pred[pred==3] = 32
        pred[pred==4] = 64
        pred[pred==5] = 128
        save(pred,os.path.join(subj_path,f'{config.model}-{config.mask}.img.gz'),hdr=hdr)

        break