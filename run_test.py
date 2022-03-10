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
from utils.DCM2IMG import DCMtoVidaCT
import SimpleITK as sitk
from medpy.io import load, save
sitk.ProcessObject_SetGlobalWarningDisplay(False)

# parser = argparse.ArgumentParser(description='segmentor')
# parser.add_argument('--mask', default='lobe', type=str, help='[airway, vessels, lung, lobe]')
# parser.add_argument('--model', default='ZUNet', type=str, help='[UNet, ZUNet]')
# # parser.add_argument('--subj_path', default='', type=str, help='Subject path, ex) VIDA_*/24')
# parser.add_argument('--in_file_path',
#     default='D:/ENV18PM/ENV18PM_ProjSubjList_IN1_test.in',
#     type=str,
#     help='path to *.in')
# parser.add_argument('--parameter_path',
#     default="RESULTS\lobe\lobe_ZUNet_n0_20220127\lobe_ZUNet_n0_29.pth",
#     type=str,
#     help='path to *.pth')

# args = parser.parse_args()
def get_config():
    config = wandb.config
    # ENV
    config.data_path = os.getenv("VIDA_PATH")
    # config.in_file_path = 'D:/ENV18PM/ENV18PM_ProjSubjList_IN1_test.in'
    config.in_file_path = 'D:/ENV18PM/ENV18PM_ProjSubjList_IN0_valid.in'

    config.parameter_path = "RESULTS\lobe\lobe_ZUNet_n0_20220127\lobe_ZUNet_n0_29.pth"
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.mask = 'lobe' # 'airway', 'lung', 'lobe'
    config.model = 'ZUNet'
    config.num_c = 6
    
    if config.model == 'ZUNet':
        config.Z = True
    else:
        config.Z = False
    config.in_c = 4
    
    return config


load_dotenv()
config = get_config()

# load model
if config.Z:
    parameter_path = config.parameter_path
    model = ZUNet_v1(in_channels=config.in_c, num_c=config.num_c)
    model.load_state_dict(torch.load(parameter_path))
    model.to(config.device)
    eng = Segmentor_Z(model=model,device=config.device)
else:
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
    if not os.path.exists(img_path):
        DCMtoVidaCT(subj_path)
    if config.in_c==1:
        img, hdr = prep_test_img(img_path, multiC=False)
        pred = eng.inference(img)

    else:
        img, hdr = prep_test_img(img_path, multiC=True)
        pred = eng.inference_multiC(img)

    if config.mask == 'lobe':
        pred[pred==1] = 8
        pred[pred==2] = 16
        pred[pred==3] = 32
        pred[pred==4] = 64
        pred[pred==5] = 128
    elif config.mask == 'airway':
        pred[pred==1] = 255
        
    save(pred,os.path.join(subj_path,f'{config.model}-{config.mask}.img.gz'),hdr=hdr)
    break
