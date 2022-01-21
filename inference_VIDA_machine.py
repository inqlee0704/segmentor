# python inference_VIDA_machine.py /E/common/ImageData/DCM_20220104_PJNMRCT_JC/DCM_20200723_PJNMRCT_0001_071458_DEID_0714582_EX_only

import sys
sys.path.insert(1,'util')
from DCM2IMG import DCMtoVidaCT
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from UNet import UNet, UNet_encoder
from ZUNet_v1 import ZUNet_v1, ZUNet_v2
# from dataloader import TE_loader
import torch
import nibabel as nib
from engine import *

import SimpleITK as sitk
from medpy.io import load, save

sitk.ProcessObject_SetGlobalWarningDisplay(False)


if __name__ == "__main__":
    DCM_path = str(sys.argv[1])
    parameter_path = 'E:\common\InKyu\silicosis\RESULTS\ZUNet_lung_multiclass_n64_CE_Tversky_20220105\ZUNet_lung_multiclass_n64_CE_Tversky_27.pth'
    # parameter_path = '/home/inqlee0704/src/DL/airway/RESULTS/Recursive_UNet_v2_20201216/model.pth'
    # model = UNet()
    if not os.path.exists(os.path.join(DCM_path,'zunu_vida-ct.img')):
        print(f'No zunu_vida-ct.img file found')
        print(f'Creating ANALYZE file from DICOM. . .')
        DCMtoVidaCT(DCM_path)

    img_path = os.path.join(DCM_path,'zunu_vida-ct.img')
    image, hdr = load(img_path)
    image = (image-(np.min(image)))/((np.max(image)-(np.min(image))))
    out = []
    out.append({'image':image})
    test_data = np.array(out)
        

    model = ZUNet_v1(in_channels=1, num_c=3)
    model.load_state_dict(torch.load(parameter_path))
    DEVICE = "cuda"
    model.to(DEVICE)
    model.eval()
    eng = Segmentor_Z(model=model)
    print(test_data[0]['image'].shape)
    pred = eng.inference(test_data[0]['image'])
    pred = pred.astype(np.ubyte)
    pred[pred==1] = 20
    pred[pred==2] = 30
    
    save(pred,os.path.join(DCM_path,'ZUNU_zunet-lung.img.gz'),hdr=hdr)


    # # Augmentation
    # # transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])

    # print("Inference . . .")
    # out_dir = "data/lung_mask/ZUNet_multiC_n64_pp"
    # os.makedirs(out_dir, exist_ok=True)
    # pbar = tqdm(enumerate(test_data), total=len(test_data))
    # for i, x in pbar:
    #     pred_label = volume_inference_multiC_z(model, x["image"])
    #     # pred_label = remove_noise(pred_label, by="centroid")
    #     hdr = nib.Nifti1Header()
    #     pair_img = nib.Nifti1Pair(pred_label, np.eye(4), hdr)
    #     nib.save(
    #         pair_img,
    #         f"{out_dir}/" + str(infer_list.loc[i, "ImgDir"][-9:-7]) + ".img.gz",
    #     )
    #     # break
    #     # pred_img = nib.Nifti1Image(pred_label, affine=np.eye(4))
    #     # pred_img.to_filename('lung_mask2/'+str(infer_list.loc[i,'ImgDir'][7:9])+'.nii.gz')
