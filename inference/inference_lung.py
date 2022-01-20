import sys
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from UNet import UNet, UNet_encoder
from ZUNet_v1 import ZUNet_v1, ZUNet_v2
from dataloader import TE_loader
import torch
import nibabel as nib

import cv2

import SimpleITK as sitk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

sitk.ProcessObject_SetGlobalWarningDisplay(False)


def remove_noise(pred, by="centroid"):
    cleared = clear_border(pred)
    label_img = label(cleared)
    # z = pred.shape[2]
    if by == "area":
        areas = [r.area for r in regionprops(label_img)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_img):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        label_img[coordinates[0], coordinates[1]] = 0
    elif by == "centroid":
        rs = regionprops(label_img)
        # if len(rs)>2:
        # Remove if the centroids are not in between 25%~75%
        for r in rs:
            if np.mean(r.coords[:, 0]) <= 100 or np.mean(r.coords[:, 0]) >= 400: # x
                label_img[r.coords[:, 0], r.coords[:, 1]] = 0
            if np.mean(r.coords[:, 1]) <= 128 or np.mean(r.coords[:, 1]) >= 384: # y
                label_img[r.coords[:, 0], r.coords[:, 1]] = 0
            if r.area < 1000:
                label_img[r.coords[:, 0], r.coords[:, 1]] = 0
                # if (
                #     np.mean(r.coords[:, 2]) / z <= 0.25
                #     or np.mean(r.coords[:, 2]) / z >= 0.75
                # ):
                # label_img[r.coords[:, 0], r.coords[:, 1]] = 0

    mask = label_img > 0
    mask = mask.astype(np.uint8) * 255
    return mask


def volume_inference_multiC_z(model, volume, threshold=0.5):
    # volume: C x H x W x Z
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    slices = np.zeros((volume.shape[1:]))
    volume = np.flip(volume, axis=-1)
    for i in range(volume.shape[-1]):
        s = volume[:, :, :, i]
        s = s.astype(np.single)
        s = torch.from_numpy(s).unsqueeze(0)
        # s = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
        z = i / (volume.shape[-1] + 1)
        z = np.floor(z * 10)
        z = torch.tensor(z, dtype=torch.int64)
        pred = model(s.to(DEVICE), z.to(DEVICE))
        pred = torch.sigmoid(pred)
        pred = np.squeeze(pred.cpu().detach())
        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0

        pred = remove_noise(np.array(pred))
        slices[:, :, i] = pred
    slices = np.flip(slices, axis=-1)
    return slices


# def run_open(pred_label, kernel=[10, 10, 10]):
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (100,100))
#     open = cv2.morphologyEx(pred_label, cv2.MORPH_OPEN, kernel, iterations=1)
#     return open


if __name__ == "__main__":
    infer_path = "data/ProjSubjList.in"
    infer_list = pd.read_csv(infer_path)
    parameter_path = (
        # "/data1/inqlee0704/silicosis/RESULTS/UNet_64_20211002/lung_UNet.pth"
        # "/data1/inqlee0704/silicosis/RESULTS/ZUNet_64_lung_20211001/ZUNet_lung.pth"
        # "/data1/inqlee0704/silicosis/RESULTS/ZUNet_128_lung_20211004/ZUNet_lung.pth"
        "/data1/inqlee0704/silicosis/RESULTS/ZUNet_v1_multiC_lung_n64_20211006/ZUNet_v1_multiC_lung.pth"
    )
    # parameter_path = '/home/inqlee0704/src/DL/airway/RESULTS/Recursive_UNet_v2_20201216/model.pth'
    # model = UNet()
    model = ZUNet_v1(in_channels=3)
    model.load_state_dict(torch.load(parameter_path))
    DEVICE = "cuda"
    model.to(DEVICE)
    test_data = TE_loader(infer_list, multi_c=True)
    model.eval()

    # Augmentation
    # transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])

    print("Inference . . .")
    out_dir = "data/lung_mask/ZUNet_multiC_n64_pp"
    os.makedirs(out_dir, exist_ok=True)
    pbar = tqdm(enumerate(test_data), total=len(test_data))
    for i, x in pbar:
        pred_label = volume_inference_multiC_z(model, x["image"])
        # pred_label = remove_noise(pred_label, by="centroid")
        hdr = nib.Nifti1Header()
        pair_img = nib.Nifti1Pair(pred_label, np.eye(4), hdr)
        nib.save(
            pair_img,
            f"{out_dir}/" + str(infer_list.loc[i, "ImgDir"][-9:-7]) + ".img.gz",
        )
        # break
        # pred_img = nib.Nifti1Image(pred_label, affine=np.eye(4))
        # pred_img.to_filename('lung_mask2/'+str(infer_list.loc[i,'ImgDir'][7:9])+'.nii.gz')
