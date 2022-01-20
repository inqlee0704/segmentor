from torch import nn
import segmentation_models_pytorch as smp

"""
segmentation loss functions
- ce_loss
- dice_loss
- combo_loss
- tversky_loss
"""

def ce_loss(outputs, targets):
    # outputs: [bs, c, w, h]
    # targets: [bs, w, h]
    CE = nn.CrossEntropyLoss()
    return CE(outputs,targets)

def dice_loss(outputs, targets, binaryclass=False):
    if binaryclass:
        DiceLoss = smp.losses.DiceLoss(mode="binary")
    else:
        DiceLoss = smp.losses.DiceLoss(mode="multiclass")
    return DiceLoss(outputs, targets)

def combo_loss(outputs, targets, binaryclass=False):
    # Dice + CE
    if binaryclass:
        DiceLoss = smp.losses.DiceLoss(mode="binary")
        CE = nn.CrossEntropyLoss()
    else:
        DiceLoss = smp.losses.DiceLoss(mode="multiclass")
        # DiceLoss = smp.losses.TverskyLoss(mode="multiclass",alpha=0.3,beta=0.7)
        CE = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(outputs, targets)
    ce_loss = CE(outputs, targets)
    loss = dice_loss + ce_loss
    return loss, ce_loss, dice_loss

def tversky_loss(outputs, targets, binaryclass=False, alpha=0.3, beta=0.7):
    if binaryclass:
        tversky = smp.losses.TverskyLoss(mode="binary",alpha=alpha,beta=beta)
    else:
        tversky = smp.losses.TverskyLoss(mode="multiclass",alpha=alpha,beta=beta)
    return tversky(outputs,targets)