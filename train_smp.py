import os
import sys
from dotenv import load_dotenv
import time
import random
import wandb
import numpy as np

# model
import segmentation_models_pytorch as smp

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
from medpy.io import load, save
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')

def wandb_config():
    # project = mask
    # run_name = f"{mask}_{'ZUNet'}_n{args.n_case}"
    debug = True
    if debug:
        project = "debug"

    # wandb.init(project=project, name=run_name)
    wandb.init(project=project)
    config = wandb.config
    # ENV
    if debug:
        config.epochs = 2
        config.n_case = 0
    else:
        config.epochs = 5
        # n_case = 0 to run all cases
        config.n_case = 0

    config.save = False
    config.debug = debug
    config.data_path = os.getenv("VIDA_PATH")
    config.in_file = "ENV18PM_ProjSubjList_IN0_train_20211129.in"
    config.in_file_valid = "ENV18PM_ProjSubjList_IN0_valid_20211129.in"
    config.test_results_dir = f"RESULTS/lobe"
    # config.name = run_name
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = 'airway'
    config.mask = mask
    config.model = 'UNet'
    config.encoder = "resnet34"
    config.activation = "leakyrelu"
    config.optimizer = "adam"
    # config.scheduler = "CosineAnnealingWarmRestarts" # "ReduceLROnPlateau"
    config.loss = "Combo loss"
    config.combined_loss = True
    config.learning_rate = 0.0002
    config.train_bs = 8
    config.valid_bs = 16
    if mask == 'lobes':
        config.num_c = 6
    elif mask == 'lungs':
        config.num_c = 3
    else:
        config.num_c = 2
    
    if config.model == 'ZUNet':
        config.Z = True
    else:
        config.Z = False
        
    config.aug = True
    config.in_c = 1
    
    config.name = "smp_test"
    return config

def main():
    
    load_dotenv()
    config = wandb_config()
    scaler = amp.GradScaler()

    if config.save:
        dirname = f'{config.encoder}_{time.strftime("%Y%m%d", time.gmtime())}'
        out_dir = os.path.join(f"RESULTS/{config.mask}", dirname)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{config.name}")


    train_loader, valid_loader = prep_dataloader(config)
    model = smp.Unet(
    encoder_name=config.encoder,        
    encoder_weights="imagenet",     
    in_channels=config.in_c,                  
    classes=config.num_c,)

    criterion = combo_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    eng = Segmentor(
            model=model.to(config.device),
            optimizer=optimizer,
            loss_fn=criterion,
            # scheduler=scheduler,
            device=config.device,
            scaler=scaler,
            combined_loss=config.combined_loss,
        )

    best_loss = np.inf
    for epoch in range(config.epochs):
        trn_loss, trn_dice_loss, trn_bce_loss = eng.train(train_loader)
        val_loss, val_dice_loss, val_bce_loss = eng.evaluate(valid_loader)
        wandb.log(
                {
                    "epoch": epoch,
                    "trn_loss": trn_loss,
                    "trn_dice_loss": trn_dice_loss,
                    "trn_bce_loss": trn_bce_loss,
                    "val_loss": val_loss,
                    "val_dice_loss": val_dice_loss,
                    "val_bce_loss": val_bce_loss,
                }
            )
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
