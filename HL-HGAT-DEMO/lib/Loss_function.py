#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:07:33 2023

@author: jinghan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
class FocalLoss(nn.Module): 
    def __init__(self, alpha=0.25, gamma=2, weight=None): 
        super(FocalLoss, self).__init__() 
        self.alpha = alpha 
        self.gamma = gamma 
        self.weight = weight 
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight) 
        
    def forward(self, preds, labels): 
        logpt = -self.bce_fn(preds, labels) 
        pt = torch.exp(logpt) 
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt 
        return loss*1e4


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = logits.shape[0]
        smooth = 1
        
        probs = F.sigmoid(logits)
        m1 = probs.view(-1)
        m2 = targets.view(-1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum() + smooth) / (m1.sum() + m2.sum() + smooth)
        score = 1 - score.sum() #/ num
        return score
    
def weighted_mse_loss(y, target):
    return torch.sum(torch.exp(target.abs()) * (y-target)**2)