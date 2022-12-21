# -*- coding: utf-8 -*-
from torch import nn
import torch
import numpy as np

'''
from https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
Thinks to  @hubutui
'''


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = tuple([input.shape[0], num_classes, input.shape[1], input.shape[2],input.shape[3]])
    result = torch.zeros(shape)
    input = torch.asarray(np.expand_dims(input.cpu().numpy(), axis=1), dtype=torch.int64)
    result = result.scatter_(1, input, 1)

    return result


class DiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1e-6, p=1, reduction='mean', num_classes=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = make_one_hot(target, self.num_classes)
        target = target.view(target.shape[0], -1).to(predict.device)

        num = torch.sum(torch.mul(predict, target)) * 2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))