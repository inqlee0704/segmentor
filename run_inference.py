import os
from dotenv import load_dotenv
from tqdm.auto import tqdm
import wandb
import numpy as np
import pandas as pd
import argparse

# Custom
from networks.UNet import UNet
from networks.ZUNet_v1 import ZUNet_v1
from engine import Segmentor, Segmentor_Z
from dataloader import prep_test_img

# ML
import torch

# Others
import cv2
from skimage.measure import label
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from utils.DCM2IMG import DCMtoVidaCT
import SimpleITK as sitk
from medpy.io import load, save
sitk.ProcessObject_SetGlobalWarningDisplay(False)

parser = argparse.ArgumentParser(description='segmentor')
parser.add_argument('--mask', default='lobes', type=str, help='[airway, vessels, lung, lobes]')
parser.add_argument('--model', default='ZUNet', type=str, help='[UNet, ZUNet]')
parser.add_argument('--subj_path', default='', type=str, help='Subject path, ex) VIDA_*/24')
parser.add_argument('--in_file_path',
    default='D:/silicosis/data/TE_ProjSubjList.in',
    type=str,
    help='path to *.in')
parser.add_argument('--parameter_path',
    default="RESULTS\lobes\ZUNet.pth",
    type=str,
    help='path to *.pth')

args = parser.parse_args()

def get_chest_mask_slice(img,kernelsize=3):
    # img: [512,512]
    img[img<-1024] = -1024
    # Otsu
    thres = threshold_otsu(img)
    binary = (img>thres).astype(np.uint8)
    binary = binary*255
    # Perform opening
    kernelSize = (kernelsize,kernelsize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    binary_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Get contours
    contours, _ = cv2.findContours(binary_open, 
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key = cv2.contourArea)
    # Convert a max area contour to a chest mask
    chest_mask_slice = np.zeros((512,512))
    cv2.fillConvexPoly(chest_mask_slice,max_contour,1)
    return chest_mask_slice

def get_chest_mask_3D(img):
    # img: [512,512,z]
    chest_mask = np.zeros(img.shape)
    for i in range(img.shape[2]):
        chest_mask[:,:,i] = get_chest_mask_slice(img[:,:,i])
    return chest_mask

def pmap_smoothing_v2(pmap, clean_lung_mask, sigma=5):
    # Apply channel-wise normalized convolution
    # Input: pmap: (HxWxZxC)
    # Output: pred_smooth: (HxWxZxC)
    pmap_smooth = np.zeros_like(pmap)
    for i in range(pmap.shape[-1]):
        pmap_smooth[:,:,:,i] = gaussian_filter(pmap[:,:,:,i]*clean_lung_mask,sigma=sigma)
        weights = gaussian_filter(clean_lung_mask,sigma=sigma)
        pmap_smooth[:,:,:,i] /= weights + 0.00001
        pmap_smooth[:,:,:,i] *= clean_lung_mask

    pred_smooth = np.argmax(pmap_smooth, axis=3)
    pred_smooth = pred_smooth.astype('float64')
    return pred_smooth

def remove_noise(mask):
# Keep the second largest (the largest is background) area and remove other regions
    label_img = label(mask)
    _, counts = np.unique(label_img,return_counts=True)
    second_largest_area_i = np.argwhere(counts==np.unique(counts)[-2])
    mask = (label_img==second_largest_area_i).astype(int)
    return mask

def clean_up_lung_sagital(mask):
    # remove unattached noises in a sagital view
    lung_mask = (mask!=0).astype(int)
    map_lung = np.sum(lung_mask,axis=0)
    map_lung = (map_lung>0).astype(int)
    clean_map_lung = remove_noise(map_lung)
    clean_lung = np.dstack([clean_map_lung]*mask.shape[0])
    clean_lung = np.transpose(clean_lung,(2, 0, 1))
    clean_mask = mask*clean_lung
    return clean_mask

def run_inference(subj_path, eng, config):
        print(subj_path)
        img_path = os.path.join(subj_path,'zunu_vida-ct.img')
        if not os.path.exists(img_path):
            DCMtoVidaCT(subj_path)
        
        img, hdr =load(img_path)
        if config.in_c==1:
            singleC_img, _ = prep_test_img(img_path, multiC=False)
            pred = eng.inference(singleC_img)

        else:
            multiC_img, _ = prep_test_img(img_path, multiC=True)
            chest_mask = get_chest_mask_3D(img)
            pmap = eng.inference_pmap_multiC(multiC_img,config.num_c)
            lobe_mask = np.argmax(pmap, axis=3)
            clean_lobe_mask = chest_mask * lobe_mask
            clean_lung_mask = (clean_lobe_mask>0).astype(np.uint8)
            pred = pmap_smoothing_v2(pmap,clean_lung_mask)
        pred = clean_up_lung_sagital(pred)

        if config.mask == 'lobes':
            pred[pred==1] = 8
            pred[pred==2] = 16
            pred[pred==3] = 32
            pred[pred==4] = 64
            pred[pred==5] = 128
        elif config.mask == 'airway':
            pred[pred==1] = 255

        save_path = os.path.join(subj_path,f'{config.model}_{config.mask}.img.gz')
        print(f'save: {save_path}')
        save(pred,save_path,hdr=hdr)

def get_config():
    config = wandb.config
    # ENV
    config.in_file_path = args.in_file_path
    config.subj_path = args.subj_path
    config.parameter_path = args.parameter_path
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.mask = args.mask # 'airway', 'lung', 'lobes'
    config.model = args.model
    if args.mask == 'lobes':
        config.num_c = 6
    elif args.mask == 'lung':
        config.num_c = 3
    else:
        config.num_c = 2
    
    if config.model == 'ZUNet':
        config.Z = True
    else:
        config.Z = False
    config.in_c = 4
    
    return config


def main():
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
    
    print('-----------------------------')
    print(f'{config.model} is successfully loaded!')
    print('-----------------------------')
    print(f'Start inferencing {config.mask}:')
    # Inference
    if len(config.subj_path)>0:
        run_inference(config.subj_path, eng, config)
    else:
        infer_df = pd.read_csv(config.in_file_path, sep='\t')
        pbar = tqdm(range(len(infer_df)))
        for i in pbar:
            subj_path = infer_df.ImgDir[i]
            run_inference(subj_path, eng, config)


if __name__ == "__main__":
    main()