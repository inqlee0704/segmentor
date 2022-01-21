import os
import sys
from dotenv import load_dotenv
import time
import random
import wandb
import numpy as np

# Custom
from networks.UNet import UNet
from networks.ZUNet_v1 import ZUNet_v1
from engine import *
from dataloader import *
from losses import *

# ML
from torch.cuda import amp
import torch
from torchsummary import summary
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

# Others
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)


def get_config():
    config = wandb.config
    # ENV
    config.n_case = 0
    config.data_path = os.getenv("VIDA_PATH")
    config.in_file_infer = "ENV18PM_ProjSubjList_sillicosis_valid.in"
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.mask = 'lobe' # 'airway', 'lung', 'lobe'
    config.model = "ZUNet"

    config.Z = True
    config.num_c = 6
    config.in_c = 1

    return config


if __name__ == "__main__":
    load_dotenv()
    config = get_config()
    infer_path = "D:/ENV18PM/ENV18PM_ProjSubjList_IN0_valid_20211129.in"
    infer_list = pd.read_csv(infer_path)
    parameter_path = 'D:/segmentor/RESULTS/ZUNET_zerospadding_n32_20220121/ZUNet_zerospadding_n32_28.pth'
    model = ZUNet_v1(in_channels=config.in_c, num_c=config.num_c)
    model.load_state_dict(torch.load(parameter_path))
    DEVICE = "cuda"
    model.to(DEVICE)
    # test_data = TE_loader(infer_list, multi_c=True)
    # model.eval()

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
