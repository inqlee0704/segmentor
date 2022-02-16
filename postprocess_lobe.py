import os
import sys
from dotenv import load_dotenv
from tqdm.auto import tqdm
import numpy as np
import argparse


# Others
import SimpleITK as sitk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from medpy.io import load, save
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='segmentor')
parser.add_argument('--in_file_path',
    default='D:/silicosis/data/TE_ProjSubjList.in',
    type=str,
    help='path to *.in')
args = parser.parse_args()

def remove_noise(mask):
# Keep the second largest (the largest is background) area and remove other regions
    label_img = label(mask)
    _, counts = np.unique(label_img,return_counts=True)
    second_largest_area_i = np.argwhere(counts==np.unique(counts)[-2])
    mask = (label_img==second_largest_area_i).astype(int)
    return mask

def clean_up_lobe(mask):
    # Get mask_map for each lobe
    map_mask1 = (mask==8).astype(int)
    map_mask1 = np.sum(map_mask1,axis=2)
    map_mask1 = (map_mask1>0).astype(int)
    map_mask1 = remove_noise(map_mask1)
    map_mask1 = np.dstack([map_mask1]*mask.shape[2])
    mask1 = mask*map_mask1
    mask1 = (mask1==8).astype(int)

    map_mask2 = (mask==16).astype(int)
    map_mask2 = np.sum(map_mask2,axis=2)
    map_mask2 = (map_mask2>0).astype(int)
    map_mask2 = remove_noise(map_mask2)
    map_mask2 = np.dstack([map_mask2]*mask.shape[2])
    mask2 = mask*map_mask2
    mask2 = (mask2==16).astype(int)

    map_mask3 = (mask==32).astype(int)
    map_mask3 = np.sum(map_mask3,axis=2)
    map_mask3 = (map_mask3>0).astype(int)
    map_mask3 = remove_noise(map_mask3)
    map_mask3 = np.dstack([map_mask3]*mask.shape[2])
    mask3 = mask*map_mask3
    mask3 = (mask3==32).astype(int)

    map_mask4 = (mask==64).astype(int)
    map_mask4 = np.sum(map_mask4,axis=2)
    map_mask4 = (map_mask4>0).astype(int)
    map_mask4 = remove_noise(map_mask4)
    map_mask4 = np.dstack([map_mask4]*mask.shape[2])
    mask4 = mask*map_mask4
    mask4 = (mask4==64).astype(int)

    map_mask5 = (mask==128).astype(int)
    map_mask5 = np.sum(map_mask5,axis=2)
    map_mask5 = (map_mask5>0).astype(int)
    map_mask5 = remove_noise(map_mask5)
    map_mask5 = np.dstack([map_mask5]*mask.shape[2])
    mask5 = mask*map_mask5
    mask5 = (mask5==128).astype(int)

    # combine postprocessed masks for each lobe
    mask_pp = np.zeros((map_mask1.shape))
    mask_pp = mask1*8+mask2*16+mask3*32+mask4*64+mask5*128
    return mask_pp

def main():
    in_file_path = args.in_file_path 
    infer_df = pd.read_csv(in_file_path, sep='\t')
    pbar = tqdm(range(len(infer_df)))
    for i in pbar:
        subj_path = infer_df.ImgDir[i]
        # subj_path = f"D:\\silicosis\\data\\Turkey_dcm\\020"
        mask_path = os.path.join(subj_path,'ZUNet_in_c4-lobe.img.gz')
        save_path = os.path.join(subj_path,'ZUNet_in_c4-lobe_pp.img.gz')
        mask, hdr = load(mask_path)
        mask_pp = clean_up_lobe(mask)
        save(mask_pp,save_path,hdr=hdr)


if __name__ == "__main__":
    main()