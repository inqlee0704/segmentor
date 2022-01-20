import os
from dotenv import load_dotenv
import time
import random
import wandb
import numpy as np

# Custom
from UNet import UNet
from ZUNet_v1 import ZUNet_v1
from engine import *
from dataloader import *
from losses import *

# ML
from torch import nn
from torch.cuda import amp
import torch
from torchsummary import summary
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
import torchvision.utils as vutils
import segmentation_models_pytorch as smp

# Others
import matplotlib.pyplot as plt
from medpy.io import load
import SimpleITK as sitk
from scipy import ndimage

sitk.ProcessObject_SetGlobalWarningDisplay(False)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def wandb_config():
    project = "lobe"
    run_name = "UNet_zeropadding_n32"
    debug = True
    if debug:
        project = "debug"

    wandb.init(project=project, name=run_name)
    config = wandb.config
    # ENV
    if debug:
        config.epochs = 1
        config.n_case = 5
    else:
        config.epochs = 40
        # n_case = 0 to run all cases
        config.n_case = 32

    config.save = False
    config.debug = debug
    config.data_path = os.getenv("VIDA_PATH")
    config.in_file = "ENV18PM_ProjSubjList_IN0_train_20211129.in"
    config.in_file_valid = "ENV18PM_ProjSubjList_IN0_valid_20211129.in"
    config.test_results_dir = "RESULTS"
    config.name = run_name
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config.mask = 'airway'
    # config.mask = "lung"
    config.mask = "lobe"
    config.model = "UNet"
    config.activation = "leakyrelu"
    config.optimizer = "adam"
    config.scheduler = "CosineAnnealingWarmRestarts"
    # config.scheduler = "ReduceLROnPlateau"
    config.loss = "Combo loss"
    config.combined_loss = True

    config.learning_rate = 0.0002
    # config.learning_rate = 0.0002
    # config.learning_rate = 0.0004
    config.train_bs = 16
    config.valid_bs = 32
    config.num_c = 6
    config.aug = True
    config.Z = False
    config.in_c = 1

    return config


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prep_test_img(multiC=False):
    # Test
    test_img_path = os.getenv("TEST_IMG_PATH")
    test_img, _ = load(test_img_path)
    # test_img, _ = load("/data1/inqlee0704/silicosis/data/inputs/02_ct.hdr")
    test_img[test_img < -1024] = -1024
    if multiC:
        z_map = np.ones((512,512,test_img.shape[2],))
        for i in range(len(z_map)):
            z_map[:,:,i] = i/(len(z_map)+1)
        test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))
        test_img = test_img[None, :]
        z_map = z_map[None, :]
        test_img = np.concatenate([test_img,z_map],axis=0)
        # narrow_c = np.copy(test_img)
        # wide_c = np.copy(test_img)
        # narrow_c[narrow_c >= -500] = -500
        # wide_c[wide_c >= 300] = 300
        # test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))
        # wide_c = (wide_c - np.min(wide_c)) / (np.max(wide_c) - np.min(wide_c))
        # narrow_c = (narrow_c - np.min(narrow_c)) / (np.max(narrow_c) - np.min(narrow_c))
        # narrow_c = narrow_c[None, :]
        # wide_c = wide_c[None, :]
        # test_img = test_img[None, :]
        # test_img = np.concatenate([test_img, narrow_c, wide_c], axis=0)
    else:
        test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))
        # test_img = test_img[None, :]
    return test_img

def plot_pmap(p_map, epoch, z=151):
    p_map = np.rot90(p_map,3,[0,1])
    fig, axs = plt.subplots(1,6, figsize=(18,12))
    im1 = axs[0].imshow(p_map[:,:,z,0])
    fig.colorbar(im1, ax=axs[0], shrink=0.15)
    im2 = axs[1].imshow(p_map[:,:,z,1])
    fig.colorbar(im2, ax=axs[1], shrink=0.15)
    im3 = axs[2].imshow(p_map[:,:,z,2])
    fig.colorbar(im3, ax=axs[2], shrink=0.15)
    im4 = axs[3].imshow(p_map[:,:,z,3])
    fig.colorbar(im4, ax=axs[3], shrink=0.15)
    im5 = axs[4].imshow(p_map[:,:,z,4])
    fig.colorbar(im5, ax=axs[4], shrink=0.15)
    im6 = axs[5].imshow(p_map[:,:,z,5])
    fig.colorbar(im6, ax=axs[5], shrink=0.15)
    # turn of axies
    [ax.set_axis_off() for ax in axs.ravel()]
    return fig

def show_images(test_img, test_pred, epoch):
    test_pred[test_pred == 1] = 128
    test_pred[test_pred == 2] = 255
    test_img = torch.from_numpy(test_img)
    test_img = test_img.permute(2, 0, 1)
    test_img = test_img.unsqueeze(1)
    # test_img = test_img.permute(3, 0, 1, 2)
    # test_img = test_img[:, 0, :, :]
    # test_img = test_img.unsqueeze(1)

    test_pred = torch.from_numpy(test_pred)
    test_pred = test_pred.permute(2, 0, 1)
    test_pred = test_pred.unsqueeze(1)

    test_img_grid = vutils.make_grid(test_img)
    test_pred_grid = vutils.make_grid(test_pred)

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title(f"CT images")
    plt.imshow(test_img_grid.permute(1, 2, 0))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title(f"Lung masks at {epoch}")
    plt.imshow(test_pred_grid.permute(1, 2, 0))

    return plt


def main():
    load_dotenv()
    seed_everything()
    config = wandb_config()
    scaler = amp.GradScaler()

    if config.save:
        dirname = f'{config.name}_{time.strftime("%Y%m%d", time.gmtime())}'
        out_dir = os.path.join("RESULTS", dirname)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{config.name}")

    train_loader, valid_loader = prep_dataloader(config)
    # criterion = smp.losses.DiceLoss(mode="multiclass")
    # criterion = nn.CrossEntropyLoss()
    criterion = combo_loss
    if config.Z:
        model = ZUNet_v1(in_channels=config.in_c, num_c=config.num_c)
        model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=config.epochs, T_mult=1, eta_min=1e-8, last_epoch=-1
        )
        # scheduler = ReduceLROnPlateau(
        #     optimizer, factor=0.5, patience=5, verbose=True
        # )
        eng = Segmentor_Z(
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            scheduler=scheduler,
            device=config.device,
            scaler=scaler,
            combined_loss=config.combined_loss,
        )
    else:
        model = UNet(in_channel=config.in_c, num_c=config.num_c)
        model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=config.epochs, T_mult=1, eta_min=1e-8, last_epoch=-1
        )
        # scheduler = ReduceLROnPlateau(
        #     optimizer, factor=0.5, patience=5, verbose=True
        # )
        eng = Segmentor(
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            scheduler=scheduler,
            device=config.device,
            scaler=scaler,
            combined_loss=config.combined_loss,
        )

    # Train
    best_loss = np.inf
    test_img = prep_test_img(multiC=False)
    wandb.watch(eng.model, log="all", log_freq=10)
    for epoch in range(config.epochs):
        if config.combined_loss:
            trn_loss, trn_dice_loss, trn_bce_loss = eng.train(train_loader)
            val_loss, val_dice_loss, val_bce_loss = eng.evaluate(valid_loader)
            # test_pred = eng.inference(test_img)
            # plt = show_images(test_img, test_pred, epoch)

            test_pmap = eng.inference_pmap(test_img, n_class=config.num_c)
            plt = plot_pmap(test_pmap, epoch)
            wandb.log(
                {
                    "epoch": epoch,
                    "trn_loss": trn_loss,
                    "trn_dice_loss": trn_dice_loss,
                    "trn_bce_loss": trn_bce_loss,
                    "val_loss": val_loss,
                    "val_dice_loss": val_dice_loss,
                    "val_bce_loss": val_bce_loss,
                    "Plot": plt,
                }
            )

        else:
            trn_loss = eng.train(train_loader)
            val_loss = eng.evaluate(valid_loader)
            test_pred = eng.inference(test_img)
            plt = show_images(test_img, test_pred, epoch)
            wandb.log(
                {
                    "epoch": epoch,
                    "trn_loss": trn_loss,
                    "val_loss": val_loss,
                    "Plot": plt,
                }
            )

        # plt.close()
        if config.scheduler == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        eng.epoch += 1
        print(f"Epoch: {epoch}, train loss: {trn_loss:5f}, valid loss: {val_loss:5f}")
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Best loss: {best_loss} at Epoch: {eng.epoch}")
            if config.save:
                model_path = path + f"_{epoch}.pth"
                torch.save(model.state_dict(), model_path)
                wandb.save(path)


if __name__ == "__main__":
    main()
