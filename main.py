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
    ReduceLROnPlateau,
)

# Others
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

import argparse


parser = argparse.ArgumentParser(description='segmentor')
parser.add_argument('--mask', default='lobes', type=str, help='[airway, vessels, lung, lobes]')
parser.add_argument('--model', default='UNet', type=str, help='[UNet, ZUNet]')
parser.add_argument('--debug', default=False, type=bool, help='[True, False]')
parser.add_argument('--save', default=True, type=bool, help='[True, False]')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--train_bs', default=16, type=int, help='train batch size')
parser.add_argument('--valid_bs', default=32, type=int, help='valid batch size')
parser.add_argument('--epochs', default=50, type=int, help='train epoch')
parser.add_argument('--n_case', default=32, type=int, help='number of cases to use')

args = parser.parse_args()


def wandb_config():
    project = args.mask
    run_name = f"{args.mask}_{args.model}_n{args.n_case}"
    debug = args.debug
    if debug:
        project = "debug"

    # wandb.init(project=project, name=run_name)
    wandb.init(project=project)
    config = wandb.config
    # ENV
    if debug:
        config.epochs = 2
        config.n_case = 5
    else:
        config.epochs = args.epochs
        # n_case = 0 to run all cases
        config.n_case = args.n_case

    config.save = args.save
    config.debug = debug
    config.data_path = os.getenv("VIDA_PATH")
    config.in_file = "ENV18PM_ProjSubjList_IN0_train.in"
    config.in_file_valid = "ENV18PM_ProjSubjList_IN0_valid.in"
    config.test_results_dir = f"RESULTS/{args.mask}"
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.mask = args.mask
    config.model = args.model
    config.activation = "leakyrelu"
    config.optimizer = "adam"
    config.scheduler = "CosineAnnealingWarmRestarts" # "ReduceLROnPlateau"
    config.loss = "Combo loss"
    config.combined_loss = True
    config.learning_rate = args.lr
    config.train_bs = args.train_bs
    config.valid_bs = args.valid_bs
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

    config.aug = True
    config.in_c = 4 # 1 or 4
    config.name = run_name

    return config


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    load_dotenv()
    seed_everything()
    config = wandb_config()
    scaler = amp.GradScaler()

    if config.save:
        dirname = f'{config.name}_{time.strftime("%Y%m%d", time.gmtime())}'
        out_dir = os.path.join(f"RESULTS/{config.mask}", dirname)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{config.name}")

    if config.in_c > 1:
        train_loader, valid_loader = prep_dataloader_P_encoding(config)
    else:
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
        model = UNet(in_channels=config.in_c, num_c=config.num_c)
        model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=config.epochs, T_mult=1, eta_min=1e-8, last_epoch=-1
        )

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
    test_img_path = os.getenv("TEST_IMG_PATH")
    if config.in_c>1:
        test_img = prep_test_img(test_img_path, multiC=True)
    else:
        test_img = prep_test_img(test_img_path, multiC=False)
    wandb.watch(eng.model, log="all", log_freq=10)
    for epoch in range(config.epochs):
        if config.combined_loss:
            trn_loss, trn_dice_loss, trn_bce_loss = eng.train(train_loader)
            val_loss, val_dice_loss, val_bce_loss = eng.evaluate(valid_loader)
            # test_pred = eng.inference(test_img)
            # plt = show_images(test_img, test_pred, epoch)
            if config.in_c > 1:
                test_pmap = eng.inference_pmap_multiC(test_img, n_class=config.num_c)
            else:
                test_pmap = eng.inference_pmap(test_img, n_class=config.num_c)

            plt = plot_pmap(test_pmap, config)
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
