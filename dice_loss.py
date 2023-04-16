import tensorflow as tf
import torch


def dice_loss(y_true, y_pred, smooth=1.):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_coefficient_loss(y_true, y_pred):
    loss_background = 0.5 * dice_loss(y_true[..., 0], y_pred[..., 0])
    loss_vessel = 2.0 * dice_loss(y_true[..., 1], y_pred[..., 1])
    loss = loss_background + loss_vessel
    return loss
    # loss = 1 - dice_loss(y_true[...,0], y_pred[...,0])
    # return loss