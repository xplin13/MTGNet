import torch
import torch.nn as nn
import numpy as np

import SSIM
from sewar.full_ref import psnr, ssim
import torch.nn.functional as F
import freq_loss

def compare_psnr(x_true, x_pred):
    return psnr(x_true, x_pred)

def compare_ssim(x_true, x_pred):
    return ssim(x_true, x_pred)[0]


def p_acc(target, prediction, width_scale, height_scale, pixel_tolerances=[1, 3, 5, 10]):
    """
    Calculate the accuracy of prediction
    :param target: (N, seq_len, 2) tensor, seq_len could be 1
    :param prediction: (N, seq_len, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)

    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)

    bs_times_seqlen = target.shape[0]
    return total_correct, bs_times_seqlen


def p_acc_wo_closed_eye(target, prediction, width_scale, height_scale, pixel_tolerances=[1, 3, 5, 10]):
    """
    Calculate the accuracy of prediction, with p tolerance and only calculated on those with fully opened eyes
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor, the last dimension is whether the eye is closed
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 3)
    prediction = prediction.reshape(-1, 2)

    dis = target[:, :2] - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)
    # check if there is nan in dist
    assert torch.sum(torch.isnan(dist)) == 0

    eye_closed = target[:, 2]  # 1 is closed eye
    # get the total number frames of those with fully opened eyes
    total_open_eye_frames = torch.sum(eye_closed == 0)

    # get the indices of those with closed eyes
    eye_closed_idx = torch.where(eye_closed == 1)[0]
    dist[eye_closed_idx] = np.inf
    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)
        assert total_correct[f'p{p_tolerance}'] <= total_open_eye_frames

    return total_correct, total_open_eye_frames.item()


def px_euclidean_dist(target, prediction, width_scale, height_scale):
    """
    Calculate the total pixel euclidean distance between target and prediction
    in a batch over the sequence length
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 3)[:, :2]
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)

    total_px_euclidean_dist = torch.sum(dist)
    sample_numbers = target.shape[0]
    return total_px_euclidean_dist, sample_numbers


class weighted_MSELoss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weights = weights
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        # batch_loss = self.mseloss(inputs, targets) * self.weights
        batch_loss = self.mseloss(inputs, targets)
        if self.reduction == 'mean':
            return torch.mean(batch_loss)
        elif self.reduction == 'sum':
            return torch.sum(batch_loss)
        else:
            return batch_loss

class L1_mae_fft_HybridLoss(torch.nn.Module):
    def __init__(self, alpha=0.84):
        super(L1_mae_fft_HybridLoss, self).__init__()
        self.mae_loss = torch.nn.L1Loss()
        self.ssim_loss = SSIM.ssim
        self.fft_loss = freq_loss.FrequencyLoss()

    def forward(self, output, target):
        target_1 = F.interpolate(target, scale_factor=0.5)
        target_2 = F.interpolate(target_1, scale_factor=0.5)

        loss_mae_2 = self.mae_loss(output[0], target_2)
        loss_mae_1 = self.mae_loss(output[1], target_1)
        loss_mae_0 = self.mae_loss(output[2], target)
        loss_mae = loss_mae_0 + loss_mae_1 + loss_mae_2

        loss_ssim_2 = 1 - self.ssim_loss(output[0], target_2)
        loss_ssim_1 = 1 - self.ssim_loss(output[1], target_1)
        loss_ssim_0 = 1 - self.ssim_loss(output[2], target)
        loss_ssim = loss_ssim_0 + loss_ssim_1 + loss_ssim_2

        loss_fft_mae_2 = self.fft_loss(output[0], target_2)
        loss_fft_mae_1 = self.fft_loss(output[1], target_1)
        loss_fft_mae_0 = self.fft_loss(output[2], target)
        loss_fft_mae = loss_fft_mae_0 + loss_fft_mae_1 + loss_fft_mae_2

        return 10 * loss_mae + 1 * loss_ssim + 0.1 * loss_fft_mae


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """平滑版L2损失函数，即Charbonnier Loss"""
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.epsilon * self.epsilon))
        return loss