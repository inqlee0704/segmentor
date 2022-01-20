import os
from dotenv import load_dotenv
import time
import random
import wandb
import pandas as pd
from medpy.io import load
import numpy as np
from tqdm.auto import tqdm


import SimpleITK as sitk

sitk.ProcessObject_SetGlobalWarningDisplay(False)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def check_files(subjlist):
    subj_paths = subjlist.loc[:, "ImgDir"].values
    img_path_ = [
        os.path.join(subj_path, "zunu_vida-ct.img") for subj_path in subj_paths
    ]
    mask_path_ = [
        os.path.join(subj_path, "ZUNU_vida-airtree.img.gz") for subj_path in subj_paths
    ]
    for i in range(len(img_path_)):
        if not os.path.exists(img_path_[i]):
            print(img_path_[i], "Not exists")
        if not os.path.exists(mask_path_[i]):
            print(mask_path_[i], "Not exists")


def get_config():

    config = wandb.config
    config.data_path = os.getenv("VIDA_PATH")
    config.in_file = "ENV18PM_ProjSubjList_IN0_train_20211129.in"
    config.in_file_valid = "ENV18PM_ProjSubjList_IN0_valid_20211129.in"

    return config


if __name__ == "__main__":
    load_dotenv()
    c = get_config()

    df_train = pd.read_csv(os.path.join(c.data_path, c.in_file), sep="\t")
    df_valid = pd.read_csv(os.path.join(c.data_path, c.in_file_valid), sep="\t")

    # Train #
    pbar = tqdm(enumerate(df_train.ImgDir), total=len(df_train))
    # pbar = tqdm(enumerate(df_valid.ImgDir), total=len(df_valid))
    for i, path in pbar:
        img, _ = load(os.path.join(path, "zunu_vida-ct.img"))
        lung, _ = load(os.path.join(path, "ZUNU_vida-lung.img.gz"))
        if img.shape != lung.shape:
            print(f"Dimension not match")
            print(path)
            print(f"img: {img.shape}  | lung: {lung.shape}")
