from tqdm.auto import tqdm
import torch
from torch.cuda import amp
from sklearn import metrics
import numpy as np

from torch import nn


def get_stats(outputs, targets):
    one_hot_targets = nn.functional.one_hot(targets,num_classes=3).permute(0,3,1,2)
    outputs = nn.Softmax(dim=1)(outputs)
    outputs = torch.argmax(outputs,dim=1)
    one_hot_outputs = nn.functional.one_hot(outputs,num_classes=3).permute(0,3,1,2)
    
    batch_size, num_classes, *dims = one_hot_targets.shape
    one_hot_outputs = one_hot_outputs.view(batch_size, num_classes, -1)
    one_hot_targets = one_hot_targets.view(batch_size, num_classes, -1)
    tp = (one_hot_outputs * one_hot_targets).sum(2)
    fp = one_hot_outputs.sum(2) - tp
    fn = one_hot_targets.sum(2) - tp
    tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)
    return tp.sum(dim=0), fp.sum(dim=0), fn.sum(dim=0), tn.sum(dim=0)
    
def get_sensitivity(tp,fn): # Recall
    return tp.float()/(tp.float()+fn.float()+1e-8)

def get_specificity(tn,fp):
    return tn.float()/(tn.float()+fp.float()+1e-8)

def get_precision(tp,fp):
    return tp.float()/(tp.float()+fp.float()+1e-8)

def get_dice_score(precision, sensitivity):
    return 2 * (precision * sensitivity)/(precision + sensitivity)

def Dice3d(a, b):
    # print(f'pred shape: {a.shape}')
    # print(f'target shape: {b.shape}')
    intersection = np.sum((a != 0) * (b != 0))
    volume = np.sum(a != 0) + np.sum(b != 0)
    if volume == 0:
        return -1
    return 2.0 * float(intersection) / float(volume)


def cal_dice(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (2.0 * intersection + smooth) / (
        pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth
    )
    return loss.mean()


def cal_dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )
    return loss.mean()


def cal_loss(outputs, targets, bce_weight=0.5):
    BCE_fn = nn.BCEWithLogitsLoss()
    bce_loss = BCE_fn(outputs, targets)
    preds = torch.sigmoid(outputs)
    dice_loss = cal_dice_loss(preds, targets)
    loss = bce_loss * bce_weight + dice_loss * (1 - bce_weight)
    return loss, bce_loss, dice_loss


class Segmentor:
    def __init__(
        self,
        model,
        optimizer=None,
        loss_fn=None,
        scheduler=None,
        device=None,
        scaler=None,
        combined_loss=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.combined_loss = combined_loss
        self.device = device
        self.scaler = scaler
        self.epoch = 0

    def train(self, data_loader):
        self.model.train()
        epoch_loss = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)

        if self.combined_loss:
            epoch_dice_loss = 0
            epoch_bce_loss = 0
            for step, batch in pbar:
                self.optimizer.zero_grad()
                inputs = batch["image"].to(self.device)
                # if BCEwithLogitsLoss,
                targets = batch["seg"].to(self.device)
                # if CrossEntropyLoss,
                # targets = batch['seg'].to(self.device)
                with amp.autocast():
                    outputs = self.model(inputs)
                    loss, bce_loss, dice_loss = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step(self.epoch + step / iters)

                epoch_loss += loss.item()
                epoch_dice_loss += dice_loss.item()
                epoch_bce_loss += bce_loss.item()
                pbar.set_description(
                    f"loss:{loss:.3f}, dice loss:{dice_loss:.3f}, bce loss:{bce_loss:.3f}"
                )
            return epoch_loss / iters, epoch_dice_loss / iters, epoch_bce_loss / iters

        else:
            for step, batch in pbar:
                self.optimizer.zero_grad()
                inputs = batch["image"].to(self.device)
                # if BCEwithLogitsLoss,
                targets = batch["seg"].to(self.device)
                # if CrossEntropyLoss,
                # targets = batch['seg'].to(self.device)
                with amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step(self.epoch + step / iters)

                epoch_loss += loss.item()
                pbar.set_description(f"loss:{loss:.3f}")
            return epoch_loss / iters

    def evaluate(self, data_loader):
        self.model.eval()
        epoch_loss = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)

        if self.combined_loss:
            epoch_dice_loss = 0
            epoch_bce_loss = 0
            with torch.no_grad():
                for step, batch in pbar:
                    inputs = batch["image"].to(self.device)
                    targets = batch["seg"].to(self.device)
                    outputs = self.model(inputs)
                    loss, bce_loss, dice_loss = self.loss_fn(outputs, targets)
                    epoch_loss += loss.item()
                    epoch_dice_loss += dice_loss.item()
                    epoch_bce_loss += bce_loss.item()

                    pbar.set_description(
                        f"loss:{loss:.3f}, dice loss:{dice_loss:.3f}, bce loss:{bce_loss:.3f}"
                    )
                return (
                    epoch_loss / iters,
                    epoch_dice_loss / iters,
                    epoch_bce_loss / iters,
                )

        else:
            with torch.no_grad():
                for step, batch in pbar:
                    inputs = batch["image"].to(self.device)
                    targets = batch["seg"].to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    epoch_loss += loss.item()

                    pbar.set_description(f"loss:{loss:.3f}")
                return epoch_loss / iters

    def inference(self, img_volume):
        # img_volume: [512,512,Z]
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        pred_volume = np.zeros(img_volume.shape)
        with torch.no_grad():
            for i in range(img_volume.shape[2]):
                slice = img_volume[:, :, i]
                slice = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0)
                out = self.model(slice.to(DEVICE, dtype=torch.float))
                pred = torch.argmax(out, dim=1)
                pred = np.squeeze(pred.cpu().detach())
                pred_volume[:, :, i] = pred
            return pred_volume

    def inference_pmap(self, img_volume, n_class):
        # img_volume: [512,512,Z]
        # pred_volume: [512,512,z,n_class]
        # n_class: number of classes, for lung: 3 (left, right, background)
        # probability map
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        pred_volume = np.zeros((512,512,img_volume.shape[2],n_class))
        with torch.no_grad():
            for i in range(img_volume.shape[2]):
                slice = img_volume[:, :, i]
                slice = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0)
                out = self.model(slice.to(DEVICE, dtype=torch.float))
                p_map = nn.Softmax(dim=1)(out)
                p_map = np.squeeze(p_map.cpu().detach())
                p_map = p_map.permute(1,2,0)
    
                pred_volume[:, :, i, :] = p_map
            return pred_volume

    def inference_pmap_multiC(self, img_volume, n_class):
        # img_volume: [channel,512,512,Z]
        # pred_volume: [512,512,z,n_class]
        # n_class: number of classes, for lung: 3 (left, right, background)
        # probability map
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        pred_volume = np.zeros((512,512,img_volume.shape[-1],n_class))
        with torch.no_grad():
            for i in range(img_volume.shape[-1]):
                slice = img_volume[:, :, :, i]
                slice = torch.from_numpy(slice).unsqueeze(0)
                out = self.model(slice.to(DEVICE, dtype=torch.float))
                p_map = nn.Softmax(dim=1)(out)
                p_map = np.squeeze(p_map.cpu().detach())
                p_map = p_map.permute(1,2,0)
    
                pred_volume[:, :, i, :] = p_map
            return pred_volume

    def get_metrics(self, data_loader):
        # outputs: [B, C, W, H]
        # targets: [B, W, H]
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)
        DEVICE = 'cuda'
        with torch.no_grad():
            for step, batch in pbar:
                inputs = batch["image"].to(DEVICE)
                targets = batch["seg"].to(DEVICE)
                outputs = self.model(inputs)
                tp, fp, fn, tn = get_stats(outputs, targets)
                if step==0:
                    tps = tp
                    fps = fp
                    fns = fn
                    tns = tn
                else:
                    tps += tp
                    fps += fp
                    fns += fn
                    tns += tn

            tps = tps.cpu().detach()
            fps = fps.cpu().detach()
            fns = fns.cpu().detach()
            tns = tns.cpu().detach()
            
            sensitivity = get_sensitivity(tps,fns)
            specificity = get_specificity(tns,fps)
            precision = get_precision(tps,fps)
            dice = get_dice_score(precision, sensitivity)
            return sensitivity, specificity, dice

class Segmentor_Z:
    def __init__(
        self,
        model,
        optimizer=None,
        loss_fn=None,
        scheduler=None,
        device=None,
        scaler=None,
        combined_loss=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.combined_loss = combined_loss
        self.device = device
        self.scaler = scaler
        self.epoch = 0

    def train(self, data_loader):
        self.model.train()
        epoch_loss = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)

        if self.combined_loss:
            epoch_dice_loss = 0
            epoch_bce_loss = 0
            for step, batch in pbar:
                self.optimizer.zero_grad()
                inputs = batch["image"].to(self.device)
                z = batch["z"].to(self.device)
                # if BCEwithLogitsLoss,
                targets = batch["seg"].to(self.device)
                # if CrossEntropyLoss,
                # targets = batch['seg'].to(self.device)
                with amp.autocast():
                    outputs = self.model(inputs, z)
                    loss, bce_loss, dice_loss = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step(self.epoch + step / iters)

                epoch_loss += loss.item()
                epoch_dice_loss += dice_loss.item()
                epoch_bce_loss += bce_loss.item()
                pbar.set_description(
                    f"loss:{loss:.3f}, dice loss:{dice_loss:.3f}, bce loss:{bce_loss:.3f}"
                )
            return epoch_loss / iters, epoch_dice_loss / iters, epoch_bce_loss / iters

        else:
            for step, batch in pbar:
                self.optimizer.zero_grad()
                inputs = batch["image"].to(self.device)
                z = batch["z"].to(self.device)
                targets = batch["seg"].to(self.device)
                with amp.autocast():
                    outputs = self.model(inputs, z)
                    loss = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step(self.epoch + step / iters)

                epoch_loss += loss.item()
                pbar.set_description(f"loss:{loss:.3f}")
            return epoch_loss / iters

    def evaluate(self, data_loader):
        self.model.eval()
        epoch_loss = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)

        if self.combined_loss:
            epoch_dice_loss = 0
            epoch_bce_loss = 0
            with torch.no_grad():
                for step, batch in pbar:
                    inputs = batch["image"].to(self.device)
                    z = batch["z"].to(self.device)
                    targets = batch["seg"].to(self.device)
                    outputs = self.model(inputs, z)
                    loss, bce_loss, dice_loss = self.loss_fn(outputs, targets)
                    epoch_loss += loss.item()
                    epoch_dice_loss += dice_loss.item()
                    epoch_bce_loss += bce_loss.item()

                    pbar.set_description(
                        f"loss:{loss:.3f}, dice loss:{dice_loss:.3f}, bce loss:{bce_loss:.3f}"
                    )
                return (
                    epoch_loss / iters,
                    epoch_dice_loss / iters,
                    epoch_bce_loss / iters,
                )

        else:
            with torch.no_grad():
                for step, batch in pbar:
                    inputs = batch["image"].to(self.device)
                    z = batch["z"].to(self.device)
                    targets = batch["seg"].to(self.device)
                    outputs = self.model(inputs, z)
                    loss = self.loss_fn(outputs, targets)
                    epoch_loss += loss.item()
                    pbar.set_description(f"loss:{loss:.3f}")
                return epoch_loss / iters

    def inference(self, img_volume):
        # img_volume: [512,512,Z]
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        pred_volume = np.zeros(img_volume.shape)
        with torch.no_grad():
            for i in range(img_volume.shape[2]):
                slice = img_volume[:, :, i]
                slice = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0)
                z = i / (img_volume.shape[2] + 1)
                z = np.floor(z * 10)
                z = torch.tensor(z, dtype=torch.int64)
                out = self.model(slice.to(DEVICE, dtype=torch.float), z.to(DEVICE))
                pred = torch.argmax(out, dim=1)
                pred = np.squeeze(pred.cpu().detach())
                pred_volume[:, :, i] = pred
            return pred_volume

    def inference_pmap(self, img_volume, n_class):
        # img_volume: [512,512,Z]
        # pred_volume: [512,512,z,n_class]
        # n_class: number of classes, for lung: 3 (left, right, background)
        # probability map
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        pred_volume = np.zeros((512,512,img_volume.shape[2],n_class))
        with torch.no_grad():
            for i in range(img_volume.shape[2]):
                slice = img_volume[:, :, i]
                slice = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0)
                z = i / (img_volume.shape[2] + 1)
                z = np.floor(z * 10)
                z = torch.tensor(z, dtype=torch.int64)
                out = self.model(slice.to(DEVICE, dtype=torch.float), z.to(DEVICE))
                p_map = nn.Softmax(dim=1)(out)
                p_map = np.squeeze(p_map.cpu().detach())
                p_map = p_map.permute(1,2,0)
    
                pred_volume[:, :, i, :] = p_map
            return pred_volume

    def get_metrics(self, data_loader):
        # outputs: [B, C, W, H]
        # targets: [B, W, H]
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)
        with torch.no_grad():
            for step, batch in pbar:
                inputs = batch["image"].to(self.device)
                z = batch["z"].to(self.device)
                targets = batch["seg"].to(self.device)
                outputs = self.model(inputs, z)
                tp, fp, fn, tn = get_stats(outputs, targets)
                if step==0:
                    tps = tp
                    fps = fp
                    fns = fn
                    tns = tn
                else:
                    tps += tp
                    fps += fp
                    fns += fn
                    tns += tn

            tps = tps.cpu().detach()
            fps = fps.cpu().detach()
            fns = fns.cpu().detach()
            tns = tns.cpu().detach()
            
            sensitivity = get_sensitivity(tps,fns)
            specificity = get_specificity(tns,fps)
            precision = get_precision(tps,fps)
            dice = get_dice_score(precision, sensitivity)
            return sensitivity, specificity, dice