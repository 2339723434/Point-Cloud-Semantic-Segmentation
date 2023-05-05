"""
Author: Cong Peng
Date: 11/10/2022
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import math
import pdb
import random

from modules.repsurface_utils import UmbrellaSurfaceConstructor, SurfaceAbstractionCD, SurfaceFeaturePropagationCD
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch_scatter import scatter, scatter_softmax, scatter_sum, scatter_std, scatter_max

from modules.pointops2.functions import pointops
from .utlis import *
from .utlis import UmbrellaSurfaceConstructor, SurfaceAbstractionCD, SurfaceFeaturePropagationCD, LocalPointFusionLayer


class AllLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bl = BoundaryLoss(config.contrast, config) if 'contrast' in config else None
        self.cen = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    def forward(self, output, target, bpoint):
        loss_ce = self.cen(output, target)
        if self.contrast_head is not None:
            loss_all += self.bl(output, target, bpoint)
        return loss_all
    
class BoundaryLoss(nn.Module):
    def __init__(self, head_cfg, config):
        super().__init__()
        self.bl = self.boundaryloss
    
    def forward(output, target, bpoint):
        loss = 0
        return loss
    
    def boundaryloss(self, feat, target, bpoint):
        
        pass 
    
    

class CBFFBlock(nn.Module):
    expansion = 1
    def __init__(self, 
                 in_planes, planes, share_planes=8, 
                 nsample=16, 
                 use_xyz=False,
                 localLayer='LocalPointFusionLayer',
                 crossLayer='CrossPointFusionLayer'):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = nn.Sequential(
            globals()[localLayer](planes, planes, share_planes, nsample),
            globals()[crossLayer](in_planes, share_planes, nsample))
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes*self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        print('yes')
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b) torch.Size([31068, 3]) torch.Size([31068, 64]) torch.Size([4])
        #print(x.shape)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x = x + identity
        x = self.relu(x)
        return [p, x, o]
        

class DEBlock(nn.Module):
    expansion = 1
    def __init__(self, 
                 in_planes, planes, share_planes=8, 
                 nsample=16, 
                 use_xyz=False,
                 localLayer='LocalPointFusionLayer'):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = nn.Sequential(
            globals()[localLayer](planes, planes, share_planes, nsample))
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes*self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        print('yes')
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b) torch.Size([31068, 3]) torch.Size([31068, 64]) torch.Size([4])
        #print(x.shape)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x = x + identity
        x = self.relu(x)
        return [p, x, o]


class Model(nn.Module):
    cbffblock = CBFFBlock
    decblock = DEBlock
    def __init__(self, args):
        super().__init__()
        center_channel = 6 if args.return_polar else 3
        repsurf_in_channel = 10
        repsurf_out_channel = 10
        self.localLayer = 'LocalPointFusionLayer'
        self.crossLayer = 'CrossPointFusionLayer'
        share_planes = 8
        planes = [64, 128, 256, 512]
        dec_planes = [256, 256, 128, 128]
        blocks = [2, 2, 4, 2]
        nsample = [16, 16, 16, 16]

        self.sa1 = SurfaceAbstractionCD(4, 32, args.in_channel + repsurf_out_channel, center_channel, [32, 32, 64],
                                        True, args.return_polar, num_sector=4)                                                                        
        self.sa2 = SurfaceAbstractionCD(4, 32, 64 + repsurf_out_channel, center_channel, [64, 64, 128],
                                        True, args.return_polar)
        self.sa3 = SurfaceAbstractionCD(4, 32, 128 + repsurf_out_channel, center_channel, [128, 128, 256],
                                        True, args.return_polar)
        self.sa4 = SurfaceAbstractionCD(4, 32, 256 + repsurf_out_channel, center_channel, [256, 256, 512],
                                        True, args.return_polar)
                                        
        self.enc1 = self._make_encoder(planes[0], blocks[0], share_planes, nsample=nsample[0])
        self.enc2 = self._make_encoder(planes[1], blocks[1], share_planes, nsample=nsample[1])
        self.enc3 = self._make_encoder(planes[2], blocks[2], share_planes, nsample=nsample[2])
        self.enc4 = self._make_encoder(planes[3], blocks[3], share_planes, nsample=nsample[3])                                
                                        

        self.fp4 = SurfaceFeaturePropagationCD(512, 256, [256, 256])
        self.fp3 = SurfaceFeaturePropagationCD(256, 128, [256, 256])
        self.fp2 = SurfaceFeaturePropagationCD(256, 64, [256, 128])
        self.fp1 = SurfaceFeaturePropagationCD(128, None, [128, 128, 128])
        
        #self.dec1 = self._make_decoder(dec_planes[0], blocks[0], share_planes, nsample=nsample[0])
        #self.dec2 = self._make_decoder(dec_planes[1], blocks[1], share_planes, nsample=nsample[1])
        #self.dec3 = self._make_decoder(dec_planes[2], blocks[2], share_planes, nsample=nsample[2])
        #self.dec4 = self._make_decoder(dec_planes[3], blocks[3], share_planes, nsample=nsample[3])

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, args.num_class),
        )

        self.surface_constructor = UmbrellaSurfaceConstructor(args.group_size + 1, repsurf_in_channel, repsurf_out_channel)
        
        
    def _make_encoder(self, in_planes, blocks, share_planes, nsample):
        layers = []
        self.in_planes = in_planes
        for _ in range(1, blocks):
            layers.append(self.cbffblock(
                in_planes=self.in_planes, 
                planes=self.in_planes, 
                share_planes=share_planes,
                nsample=nsample,
                localLayer=self.localLayer,
                crossLayer=self.crossLayer))
        
        return nn.Sequential(*layers)
    
    def _make_decoder(self, in_planes, blocks, share_planes, nsample):
        layers = []
        self.in_planes = in_planes
        for _ in range(1, blocks):
            layers.append(self.decblock(
                in_planes=self.in_planes, 
                planes=self.in_planes, 
                share_planes=share_planes,
                nsample=nsample,
                localLayer=self.localLayer))
        
        return nn.Sequential(*layers)

    def forward(self, pos_feat_off0):
        print('****************NetWork Forward***************')
        #print(pos_feat_off0[0].shape, pos_feat_off0[1].shape, pos_feat_off0[2])
        #exit(0)
        pos_nor_feat_off0 = [
            pos_feat_off0[0],
            self.surface_constructor(pos_feat_off0[0], pos_feat_off0[2]),
            torch.cat([pos_feat_off0[0], pos_feat_off0[1]], 1),
            pos_feat_off0[2]
        ]
        #torch.Size([213099, 3]) torch.Size([213099, 10]) torch.Size([213099, 6])
        #print(pos_nor_feat_off0[0].shape,pos_nor_feat_off0[1].shape,pos_nor_feat_off0[2].shape,pos_nor_feat_off0[3])
        #exit(0)
        
        
        pos_nor_feat_off1 = self.sa1(pos_nor_feat_off0) #torch.Size([53301, 64])
        pos_nor_feat_off1[0], pos_nor_feat_off1[2],pos_nor_feat_off1[3] = self.enc1([pos_nor_feat_off1[0], pos_nor_feat_off1[2], pos_nor_feat_off1[3]])
        #pos_nor_feat_off1 = self.enc1([pos_nor_feat_off1[0], pos_nor_feat_off1[2], pos_nor_feat_off1[3]])
        
        
        pos_nor_feat_off2 = self.sa2(pos_nor_feat_off1) #torch.Size([13324, 128])
        pos_nor_feat_off2[0], pos_nor_feat_off2[2],pos_nor_feat_off2[3] = self.enc2([pos_nor_feat_off2[0], pos_nor_feat_off2[2], pos_nor_feat_off2[3]])
        
        pos_nor_feat_off3 = self.sa3(pos_nor_feat_off2) #torch.Size([3330, 256])
        pos_nor_feat_off3[0], pos_nor_feat_off3[2],pos_nor_feat_off3[3] = self.enc3([pos_nor_feat_off3[0], pos_nor_feat_off3[2], pos_nor_feat_off3[3]])
        
        pos_nor_feat_off4 = self.sa4(pos_nor_feat_off3) #torch.Size([831, 512])
        pos_nor_feat_off4[0], pos_nor_feat_off4[2],pos_nor_feat_off4[3] = self.enc4([pos_nor_feat_off4[0], pos_nor_feat_off4[2], pos_nor_feat_off4[3]])
        
        del pos_nor_feat_off0[1], pos_nor_feat_off1[1], pos_nor_feat_off2[1], pos_nor_feat_off3[1], pos_nor_feat_off4[1]
        pos_nor_feat_off3[1] = self.fp4(pos_nor_feat_off3, pos_nor_feat_off4) #torch.Size([3330, 256])
        #pos_nor_feat_off3[0],pos_nor_feat_off3[1],pos_nor_feat_off3[2] = self.dec1([pos_nor_feat_off3[0], pos_nor_feat_off3[1], pos_nor_feat_off3[2]])
        
        pos_nor_feat_off2[1] = self.fp3(pos_nor_feat_off2, pos_nor_feat_off3) #torch.Size([13324, 256])
        #pos_nor_feat_off2[0],pos_nor_feat_off2[1],pos_nor_feat_off2[2] = self.dec2([pos_nor_feat_off2[0], pos_nor_feat_off2[1], pos_nor_feat_off2[2]])
        
        pos_nor_feat_off1[1] = self.fp2(pos_nor_feat_off1, pos_nor_feat_off2) #torch.Size([53301, 128])
        #pos_nor_feat_off1[0],pos_nor_feat_off1[1],pos_nor_feat_off1[2] = self.dec3([pos_nor_feat_off1[0], pos_nor_feat_off1[1], pos_nor_feat_off1[2]])
        
        pos_nor_feat_off0[1] = self.fp1([pos_nor_feat_off0[0], None, pos_nor_feat_off0[2]], pos_nor_feat_off1) #torch.Size([213209, 128])
        #pos_nor_feat_off0[0],pos_nor_feat_off0[1],pos_nor_feat_off0[2] = self.dec4([pos_nor_feat_off0[0], pos_nor_feat_off0[1], pos_nor_feat_off0[2]])
        #exit(0)
        
        #del pos_nor_feat_off1, pos_nor_feat_off2, pos_nor_feat_off3, pos_nor_feat_off4, pos_nor_feat_off0[0]

        feature = self.classifier(pos_nor_feat_off0[1])

        return feature
