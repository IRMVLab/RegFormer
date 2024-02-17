# -*- coding:UTF-8 -*-

import os
import sys
import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
from pointnet2 import pointnet2_utils

sys.path.append('ops_pytorch/fused_conv_select_k')
sys.path.append('ops_pytorch/fused_conv_random_k')

from fused_conv_random_k import fused_conv_random_k
from fused_conv_select_k import fused_conv_select_k

LEAKY_RATE = 0.1
use_bn = False




def get_hw_idx(B, out_H, out_W, stride_H = 1, stride_W = 1):

    H_idx = torch.reshape(torch.arange(0, out_H * stride_H, stride_H), [1, -1, 1, 1]).expand(B, out_H, out_W, 1)
    W_idx = torch.reshape(torch.arange(0, out_W * stride_W, stride_W), [1, 1, -1, 1]).expand(B, out_H, out_W, 1)

    idx_n2 = torch.cat([H_idx, W_idx], dim = -1).reshape(B, -1, 2)


    return idx_n2


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_activation=True,
                 use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if use_activation:
            relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        else:
            relu = nn.Identity()

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.composed_module(x)
        x = x.permute(0, 2, 1)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=[1, 1], bn=False, activation_fn=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = bn
        self.activation_fn = activation_fn

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if bn:
            self.bn_linear = nn.BatchNorm2d(out_channels)

        if activation_fn:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x (b,n,s,c)
        # print('x is ')
        # print('x: ', x.device)
        x = x.permute(0, 3, 2, 1)  # (b,c,s,n)
        # print(self.conv)
        outputs = self.conv(x)
        # print('self conv has be carried out')
        if self.bn:
            outputs = self.bn_linear(outputs)

        if self.activation_fn:
            outputs = self.relu(outputs)

        outputs = outputs.permute(0, 3, 2, 1)  # (b,n,s,c)
        return outputs

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm?    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def calc_cosine_similarity(desc1, desc2):
    '''
    Input:
        desc1: [B,N,*,C]
        desc2: [B,N,*,C]
    Ret:
        similarity: [B,N,*]
    '''
    inner_product = torch.sum(torch.mul(desc1, desc2), dim=-1, keepdim=False)
    norm_1 = torch.norm(desc1, dim=-1, keepdim=False)
    norm_2 = torch.norm(desc2, dim=-1, keepdim=False)
    similarity = inner_product/(torch.mul(norm_1, norm_2)+1e-6)
    return similarity

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points


def grouping(feature, K, src_xyz, q_xyz, use_xyz=False):
    '''
    Input:
        feature: (batch_size, ndataset, c)
        K: neighbor size
        src_xyz: original point xyz (batch_size, ndataset, 3)
        q_xyz: query point xyz (batch_size, npoint, 3)
    Return:
        grouped_xyz: (batch_size, npoint, K,3)
        xyz_diff: (batch_size, npoint,K, 3)
        new_points: (batch_size, npoint,K, c+3) if use_xyz else (batch_size, npoint,K, c)
        point_indices: (batch_size, npoint, K)
    '''

    q_xyz  = q_xyz.contiguous()
    src_xyz = src_xyz.contiguous()

    point_indices = knn_point(K,src_xyz,q_xyz)   # (batch_size, npoint, K)

    grouped_xyz = index_points_group(src_xyz,point_indices)   # (batch_size, npoint, K,3)
    xyz_diff = grouped_xyz - (q_xyz.unsqueeze(2)).repeat(1, 1, K, 1)  #  (batch_size, npoint,K, 3)

    grouped_feature = index_points_group(feature, point_indices) #(batch_size, npoint, K,c)
    if use_xyz:
        new_points = torch.cat([xyz_diff, grouped_feature], dim=-1)  # (batch_size, npoint,K, c+3)
    else:
        new_points = grouped_feature  #(batch_size, npoint, K,c)

    return grouped_xyz, xyz_diff, new_points, point_indices


class PointNetSaModule(nn.Module):

    def __init__(self, batch_size, K_sample, kernel_size, H, W, stride_H, stride_W, distance, in_channels, mlp, is_training, bn_decay,
                           bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):

        super(PointNetSaModule,self).__init__()

        self.batch_size = batch_size
        self.K_sample = K_sample
        self.kernel_size = kernel_size
        self.H = H; self.W = W
        self.stride_H = stride_H; self.stride_W = stride_W
        self.distance = distance
        self.in_channels = in_channels + 3
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.use_xyz = use_xyz
        self.use_nchw = use_nchw
        self.mlp_convs = nn.ModuleList()

        for i,num_out_channel in enumerate(mlp):
            self.mlp_convs.append(Conv2d(self.in_channels,num_out_channel,[1,1],stride=[1,1],bn=bn))
            self.in_channels = num_out_channel

    def forward(self, xyz_proj, points_proj, xyz_sampled_proj):
        
        self.idx_n2 = get_hw_idx(self.batch_size, out_H = self.H, out_W = self.W, stride_H = self.stride_H, stride_W = self.stride_W) ## b -1 2

        B = xyz_proj.shape[0]
        H = xyz_proj.shape[1]
        W = xyz_proj.shape[2]
        C = points_proj.shape[3]

        h = xyz_sampled_proj.shape[1]
        w = xyz_sampled_proj.shape[2]

        kernel_total = self.kernel_size[0] * self.kernel_size[1]
    
        n_sampled = self.idx_n2.shape[1]    #  n_sampled = h*w

        random_HW = (torch.arange(self.kernel_size[0] * self.kernel_size[1])).int().cuda()
        
        ##randperm 
        
        #################   fused_conv   default  padding = 'same'

        select_b_idx = torch.zeros(B, n_sampled, self.K_sample, 1).cuda().long().detach()       # (B, n_sampled, K_sampled, 1)
        select_h_idx = torch.zeros(B, n_sampled, self.K_sample, 1).cuda().long().detach()
        select_w_idx = torch.zeros(B, n_sampled, self.K_sample, 1).cuda().long().detach()

        valid_idx = torch.zeros(B, n_sampled, kernel_total, 1).cuda().float().detach()                  # (B, n_sampled, H*W, 1)    
        valid_in_dis_idx = torch.zeros(B, n_sampled, kernel_total, 1).cuda().float().detach()
        valid_mask = torch.zeros(B, n_sampled, self.K_sample, 1).cuda().float().detach()     # (B, n_sampled, K_sampled, 1)

        idx_n2_part = self.idx_n2.cuda().int().contiguous()   # (B N 2)
        
        with torch.no_grad():
        # Sample n' points from input n points 
            select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, valid_mask = \
            fused_conv_random_k(xyz_proj.contiguous(), xyz_proj.contiguous(), idx_n2_part, random_HW, H, W, n_sampled, self.kernel_size[0], self.kernel_size[1], self.K_sample, 0, self.distance,\
                1, 1, select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, valid_mask, H, W)
            
        neighbor_idx = select_h_idx * W + select_w_idx    
        neighbor_idx = neighbor_idx.reshape(B, -1)
        xyz_bn3 = xyz_proj.reshape(B, -1, 3)
        points_bn3 = points_proj.reshape(B, -1, C)
        
        new_xyz_group = torch.gather(xyz_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, 3))
        new_points_group = torch.gather(points_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, C))
        
        
        # Directly output 3D coordinate xyz of KNN of sampled point p' with mask
        # new_xyz_group = xyz_proj[select_b_idx, select_h_idx, select_w_idx, : ]          # (B, n_sampled, K_sampled, 1, 3) 
        new_xyz_group = new_xyz_group.reshape(B,n_sampled,self.K_sample,3)              # (B, n_sampled, K_sampled, 3)
        new_xyz_group = new_xyz_group * valid_mask                                      # (B, n_sampled, K_sampled, 3)

        # new_points_group = points_proj[select_b_idx, select_h_idx, select_w_idx, : ]    # (B, n_sampled, K_sampled, 1, 3) 
        new_points_group = new_points_group.reshape(B,n_sampled,self.K_sample,C)        # (B, n_sampled, K_sampled, C)
        new_points_group = new_points_group * valid_mask                               # (B, n_sampled, K_sampled, C)
        
        #  Directly output 3D coordinate xyz' of sampled point p' with added values
        new_xyz_proj = xyz_sampled_proj                              # (B, h, w, 3) 
       
        new_xyz = new_xyz_proj.reshape(B,-1,3)                                          # B, n_sampled, 3
        new_xyz_expand = torch.unsqueeze(new_xyz,2).expand(B, h*w ,self.K_sample,3)         # B, n_sampled, K_sampled, 3
        
        xyz_diff = new_xyz_group - new_xyz_expand                                           # (B, n_sampled, K_sampled, 3)
        new_points_group_concat = torch.cat([xyz_diff, new_points_group], dim = -1)        # (B, n_sampled, K_sampled, C+3) 
        
        # Point Feature Embedding -- shared MLP  
        for i,conv in enumerate(self.mlp_convs):
            new_points_group_concat = conv(new_points_group_concat)                         # (B, n_sampled, K_sampled, mlp[-1])              

        # Pooling in Local Regions -- max pooling (gather K points feature to 1)
        if self.pooling=='max':
            new_points_group_concat = torch.max(new_points_group_concat, dim=2, keepdim=True)[0]      # (B, n_sampled, 1, mlp[-1])

        elif self.pooling=='avg':
            new_points_group_concat = torch.mean(new_points_group_concat, dim=2, keepdim=True)     # (B, n_sampled, 1, mlp[-1])

        points_down_sample = torch.squeeze(new_points_group_concat, 2)           # (B, n_sampled, mlp2[-1])
        points_down_sample_proj = torch.reshape(points_down_sample, [B, h, w, -1])           # (B, n_sampled, mlp2[-1])
        
        return points_down_sample, points_down_sample_proj


class All2AllCostVolume(nn.Module):
    def __init__(self, radius, nsample, nsample_q, in_channels, mlp1, mlp2, is_training, bn_decay, bn=True,
                 pooling='max', knn=True, corr_func='elementwise_product', use_neighbor=True, use_sim=True):
        super(All2AllCostVolume, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.nsample_q = nsample_q
        self.in_channels = 3 * in_channels + 10
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.corr_func = corr_func
        self.use_neighbor = use_neighbor
        self.use_sim = use_sim

        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_convs_new = nn.ModuleList()

        for i, num_out_channel in enumerate(mlp1):
            self.mlp1_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.pi_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        self.in_channels = 2 * mlp1[-1]
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.pc_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        self.in_channels = 2 * mlp1[-1] + in_channels
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs_new.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.pi_reverse_encoding = Conv2d(in_channels, in_channels, [1, 1], stride=[1, 1], bn=True)

        ########neighbor encoding

        out_channels = [in_channels * 2 + 14, in_channels * 2, in_channels * 2, in_channels]
        layers = []
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], kernel_size=1, bias=False),
                       nn.BatchNorm2d(out_channels[i]),
                       nn.ReLU()]
        self.convs_1 = nn.Sequential(*layers)

        out_channels_nbr = [in_channels + 4, in_channels, in_channels, in_channels]
        self_layers = []
        for i in range(1, len(out_channels_nbr)):
            self_layers += [nn.Conv2d(out_channels_nbr[i - 1], out_channels_nbr[i], kernel_size=1, bias=False),
                            nn.BatchNorm2d(out_channels_nbr[i]),
                            nn.ReLU()]
        self.convs_2 = nn.Sequential(*self_layers)

    def forward(self, warped_xyz, warped_points, f2_xyz, f2_points):
        '''
            Input:
                warped_xyz: (b,npoint,3)
                warped_points: (b,npoint,c)
                f2_xyz:  (b,ndataset,3)
                f2_points: (b,ndataset,c)

            Output:
                pc_feat1_new: batch_size, npoints, mlp2[-1]
            '''
        _, npoints, _ = warped_xyz.shape

        # src_desc = warped_points.permute(0, 2, 1).contiguous()
        # dst_desc = f2_points.permute(0, 2, 1).contiguous()
        dst_knn_xyz, _, dst_knn_desc, src_knn_idx = grouping(f2_points, self.nsample_q, f2_xyz, warped_xyz)
        # [B,N1,N2,3] [B,N1,N2,C] [B,N1,N2]

        src_xyz_expand = warped_xyz.unsqueeze(2).repeat(1, 1, self.nsample_q, 1)
        src_desc_expand = warped_points.unsqueeze(2).repeat(1, 1, self.nsample_q, 1)  # [B,N1,N2,C]
        src_rela_xyz = dst_knn_xyz - src_xyz_expand  # [B,N1,N2,3]
        src_rela_dist = torch.norm(src_rela_xyz, dim=-1, keepdim=True)  # [B,N1,N2,1]


        if self.use_sim:
            # construct original similarity features
            dst_desc_expand_N = f2_points.unsqueeze(2).repeat(1, 1, warped_xyz.shape[1], 1)  # [B,N2,N1,C]
            src_desc_expand_N = warped_points.unsqueeze(1).repeat(1, f2_xyz.shape[1], 1, 1)  # [B,N2,N1,C]

            dst_src_cos = calc_cosine_similarity(dst_desc_expand_N, src_desc_expand_N)  # [B,N2,N1]
            dst_src_cos_max = torch.max(dst_src_cos, dim=2, keepdim=True)[0]  # [B,N2,1]
            dst_src_cos_norm = dst_src_cos / (dst_src_cos_max + 1e-6)  # [B,N2,N1]

            src_dst_cos = dst_src_cos.permute(0, 2, 1)  # [B,N1,N2]
            src_dst_cos_max = torch.max(src_dst_cos, dim=2, keepdim=True)[0]  # [B,N1,1]
            src_dst_cos_norm = src_dst_cos / (src_dst_cos_max + 1e-6)  # [B,N1,N2]

            dst_src_cos_knn = index_points_group(dst_src_cos_norm, src_knn_idx)  # [B,N1,N2,N1]
            dst_src_cos = torch.zeros(dst_knn_xyz.shape[0], dst_knn_xyz.shape[1], dst_knn_xyz.shape[2]).cuda()  # [B,N1,k]
            for i in range(warped_xyz.shape[1]):
                dst_src_cos[:, i, :] = dst_src_cos_knn[:, i, :, i]

            src_dst_cos_knn = index_points_group(src_dst_cos_norm.permute(0, 2, 1), src_knn_idx) # [B,N1,N2,N1]
            src_dst_cos = torch.zeros(dst_knn_xyz.shape[0], dst_knn_xyz.shape[1], dst_knn_xyz.shape[2]).cuda()  # [B,N1,N2]
            for i in range(warped_xyz.shape[1]):
                src_dst_cos[:, i, :] = src_dst_cos_knn[:, i, :, i]
        ############################################################################################################
        if self.use_neighbor:
            src_xyz_grouped, _, src_nbr_knn_feats, idx = grouping(warped_points, self.nsample, warped_xyz, warped_xyz)
            src_xyz_expanded = (torch.unsqueeze(warped_xyz, 2)).repeat([1, 1, self.nsample, 1])  # batch_size, npoints, nsample, 3
            src_nbr_knn_rela_xyz = src_xyz_grouped - src_xyz_expanded
            src_nbr_knn_rela_dist = torch.norm(src_nbr_knn_rela_xyz, dim=-1, keepdim=True)  # [B,N,k]
            src_nbr_feats = torch.cat([src_nbr_knn_feats, src_nbr_knn_rela_xyz, src_nbr_knn_rela_dist], dim=-1) #[B,N,K,4]

            dst_xyz_grouped, _, dst_nbr_knn_feats, idx = grouping(f2_points, self.nsample, f2_xyz, f2_xyz)
            dst_xyz_expanded = (torch.unsqueeze(f2_xyz, 2)).repeat([1, 1, self.nsample, 1])  # batch_size, npoints, nsample, 3
            dst_nbr_knn_rela_xyz = dst_xyz_grouped - dst_xyz_expanded
            dst_nbr_knn_rela_dist = torch.norm(dst_nbr_knn_rela_xyz, dim=-1, keepdim=True)  # [B,N,k]
            dst_nbr_feats = torch.cat([dst_nbr_knn_feats, dst_nbr_knn_rela_xyz, dst_nbr_knn_rela_dist], dim=-1)

            src_nbr_weights = self.convs_2(src_nbr_feats.permute(0, 3, 1, 2).contiguous())
            src_nbr_weights = torch.max(src_nbr_weights, dim=1, keepdim=False)[0]
            src_nbr_weights = F.softmax(src_nbr_weights, dim=-1)
            src_nbr_desc = torch.sum(torch.mul(src_nbr_knn_feats, src_nbr_weights.unsqueeze(-1)), dim=2, keepdim=False)

            dst_nbr_weights = self.convs_2(dst_nbr_feats.permute(0, 3, 1, 2).contiguous())
            dst_nbr_weights = torch.max(dst_nbr_weights, dim=1, keepdim=False)[0]
            dst_nbr_weights = F.softmax(dst_nbr_weights, dim=-1)
            dst_nbr_desc = torch.sum(torch.mul(dst_nbr_knn_feats, dst_nbr_weights.unsqueeze(-1)), dim=2, keepdim=False)

            dst_nbr_desc_expand_N = dst_nbr_desc.unsqueeze(2).repeat(1, 1, warped_xyz.shape[1], 1)  # [B,N2,N1,C]
            src_nbr_desc_expand_N = src_nbr_desc.unsqueeze(1).repeat(1, f2_xyz.shape[1], 1, 1)  # [B,N2,N1,C]

            dst_src_nbr_cos = calc_cosine_similarity(dst_nbr_desc_expand_N, src_nbr_desc_expand_N)  # [B,N2,N1]
            dst_src_nbr_cos_max = torch.max(dst_src_nbr_cos, dim=2, keepdim=True)[0]  # [B,N2,1]
            dst_src_nbr_cos_norm = dst_src_nbr_cos / (dst_src_nbr_cos_max + 1e-6)  # [B,N2,N1]

            src_dst_nbr_cos = dst_src_nbr_cos.permute(0, 2, 1)  # [B,N1,N2]
            src_dst_nbr_cos_max = torch.max(src_dst_nbr_cos, dim=2, keepdim=True)[0]  # [B,N1,1]
            src_dst_nbr_cos_norm = src_dst_nbr_cos / (src_dst_nbr_cos_max + 1e-6)  # [B,N1,N2]
            # print(src_idx.shape)
            # print(dst_src_nbr_cos_norm.shape)

            dst_src_nbr_cos_knn = index_points_group(dst_src_nbr_cos_norm, src_knn_idx)
            dst_src_nbr_cos = torch.zeros(dst_knn_xyz.shape[0], dst_knn_xyz.shape[1], dst_knn_xyz.shape[2]).to(dst_knn_xyz.cuda())
            # print(dst_src_nbr_cos.shape)
            # print(dst_src_nbr_cos_knn.shape)
            for i in range(warped_xyz.shape[1]):
                dst_src_nbr_cos[:, i, :] = dst_src_nbr_cos_knn[:, i, :, i]

            src_dst_nbr_cos_knn = index_points_group(src_dst_nbr_cos_norm.permute(0, 2, 1), src_knn_idx)
            src_dst_nbr_cos = torch.zeros(dst_knn_xyz.shape[0], dst_knn_xyz.shape[1], dst_knn_xyz.shape[2]).to(dst_knn_xyz.cuda())
            for i in range(warped_xyz.shape[1]):
                src_dst_nbr_cos[:, i, :] = src_dst_nbr_cos_knn[:, i, :, i]

        ############################
        geom_feats = torch.cat([src_rela_xyz, src_rela_dist, src_xyz_expand, dst_knn_xyz], dim=-1) # [B,N1,N2,10]
        desc_feats = torch.cat([src_desc_expand, dst_knn_desc], dim=-1) # [B,N1,N2,2C]
        if self.use_sim and self.use_neighbor:
            similarity_feats = torch.cat([src_dst_cos.unsqueeze(-1), dst_src_cos.unsqueeze(-1), \
                                          src_dst_nbr_cos.unsqueeze(-1), dst_src_nbr_cos.unsqueeze(-1)], dim=-1) # [B,N1,N2,4]
        elif self.use_sim:
            similarity_feats = torch.cat([src_dst_cos.unsqueeze(-1), dst_src_cos.unsqueeze(-1)], dim=-1)
        elif self.use_neighbor:
            similarity_feats = torch.cat([src_dst_nbr_cos.unsqueeze(-1), dst_src_nbr_cos.unsqueeze(-1)], dim=-1)
        else:
            similarity_feats = None

        feats = torch.cat([geom_feats, desc_feats, similarity_feats], dim=-1)
        feats = self.convs_1(feats.permute(0, 3, 1, 2)) #2c+14_>c
        attentive_weights = torch.max(feats, dim=1)[0]
        attentive_weights = F.softmax(attentive_weights, dim=-1)  # [B,N1,N2]
        attentive_feats = torch.sum(torch.mul(attentive_weights.unsqueeze(1), feats), dim=-1, keepdim=False).permute(0, 2, 1)  # [B,N1,C]

        return attentive_feats
   
class cost_volume(nn.Module):
    def __init__(self, batch_size, kernel_size1, kernel_size2, nsample, nsample_q, \
        H, W, stride_H, stride_W, distance, in_channels, mlp1, mlp2, is_training, bn_decay, bn=True, pooling='max', knn=True, \
        corr_func='elementwise_product', distance2 = 100 ):
        super(cost_volume,self).__init__()

        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.nsample = nsample
        self.nsample_q = nsample_q
        self.in_channels = in_channels[0] + in_channels[1] + 10
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.corr_func = corr_func
        self.distance1 = distance
        self.distance2 = distance2
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_convs_new = nn.ModuleList()

        self.idx_n2 = get_hw_idx(batch_size, H, W, stride_H, stride_W)

        for i, num_out_channel in enumerate(mlp1):
            self.mlp1_convs.append(Conv2d(self.in_channels,num_out_channel,[1,1],stride=[1,1], bn=self.bn))
            self.in_channels = num_out_channel

        self.pi_encoding = Conv2d(10,mlp1[-1],[1,1],stride=[1,1], bn=self.bn)

        self.in_channels = 2*mlp1[-1]     
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs.append(Conv2d(self.in_channels,num_out_channel,[1,1],stride=[1,1], bn=self.bn))
            self.in_channels = num_out_channel

        self.pc_encoding = Conv2d(10,mlp1[-1], [1,1],stride=[1,1], bn=self.bn)

        self.in_channels = 2 * mlp1[-1] + in_channels[1]
        for j,num_out_channel in enumerate(mlp2):
            self.mlp2_convs_new.append(Conv2d(self.in_channels,num_out_channel,[1,1],stride=[1,1], bn=self.bn))
            self.in_channels = num_out_channel

    def forward(self, warped_xyz1_proj, xyz2_proj, points1_proj, points2_proj):


        B = warped_xyz1_proj.shape[0]
        H = warped_xyz1_proj.shape[1]
        W = warped_xyz1_proj.shape[2]
        C = points2_proj.shape[3]

        warped_xyz1 = warped_xyz1_proj.reshape(B, -1, 3)
        points1 = points1_proj.reshape(B, -1, points1_proj.shape[-1])

        kernel_total_q = self.kernel_size2[0] * self.kernel_size2[1]

        random_HW_q = (torch.arange(0,kernel_total_q)).cuda().int()

        idx_hw = self.idx_n2.cuda().int().contiguous()

        # Initialize
        select_b_idx = torch.zeros(B, H*W, self.nsample_q, 1).cuda().long().detach()             # (B N nsample_q 1)
        select_h_idx = torch.zeros(B, H*W, self.nsample_q, 1).cuda().long().detach()
        select_w_idx = torch.zeros(B, H*W, self.nsample_q, 1).cuda().long().detach()

        valid_idx = torch.zeros(B, H*W, kernel_total_q, 1).cuda().float().detach()
        valid_in_dis_idx = torch.zeros(B, H*W, kernel_total_q, 1).cuda().float().detach()
        select_mask = torch.zeros(B, H*W, self.nsample_q, 1).cuda().float().detach()
        
        with torch.no_grad():
        # Sample QNN of (M neighbour points from sampled n points in PC1) in PC2
            select_b_idx, select_h_idx, select_w_idx, valid_idx2, valid_in_dis_idx2, valid_mask = fused_conv_select_k\
                ( warped_xyz1_proj, xyz2_proj, idx_hw, random_HW_q, H, W, H * W, self.kernel_size2[0], self.kernel_size2[1],\
                    self.nsample_q,  0, self.distance2, 1, 1, select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, select_mask, H, W)
            
        
        neighbor_idx = select_h_idx * W + select_w_idx    
        neighbor_idx = neighbor_idx.reshape(B, -1)
        xyz2_bn3 = xyz2_proj.reshape(B, -1, 3)
        points2_bn3 = points2_proj.reshape(B, -1, C)
        
        qi_xyz_grouped = torch.gather(xyz2_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, 3))
        qi_points_grouped = torch.gather(points2_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, C))
        
        

        # B N M 3 idx

        # Output 3D coordinates xyz2(Q) of QNN of M nearest neighbours of sampled n points in PC2
        # qi_xyz_grouped = xyz2_proj[qi_xyz_points_b_idx, qi_xyz_points_h_idx, qi_xyz_points_w_idx, : ]    

        qi_xyz_grouped = qi_xyz_grouped.reshape(B, H*W, self.nsample_q, 3) 
        qi_xyz_grouped = qi_xyz_grouped * (valid_mask.detach())                            # (B N nsample_q 3)
    

        # Output features f2(Q) of QNN of the M nearest neighbours of sampled n points in PC2
        # qi_points_grouped = points2_proj[qi_xyz_points_b_idx, qi_xyz_points_h_idx, qi_xyz_points_w_idx, : ]
        qi_points_grouped = qi_points_grouped.reshape(B, H*W, self.nsample_q, C)
        qi_points_grouped = qi_points_grouped * (valid_mask.detach())    
                                

        # Output 3D coordinates xyz1(M) of M nearest neighbours of sampled n points in PC1
        pi_xyz_expanded = (torch.unsqueeze(warped_xyz1, 2)).expand(B, H*W, self.nsample_q, 3)     # (B N nsample_q 3)
        
        # Output features f1(M) of M nearest neighbours of sampled n points in PC1
        pi_points_expanded = (torch.unsqueeze(points1, 2)).expand(B, H*W, self.nsample_q, points1.shape[-1])      #(B N nsample_q C)

        
        # xyz2(Q) - xyz1(M)
        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded         # (B N nsample_q 3)
        
        # Euclidean difference ---- square of xyz2(Q) - xyz1(M)
        pi_euc_diff = torch.sqrt(torch.sum(torch.square(pi_xyz_diff), dim=-1 , keepdim=True) + 1e-20 )      # (B N nsample_q 1)

        # 3D Euclidean Space Information --- (3D coordinates xyz1(M) /// 3D coordinates xyz2(Q) /// xyz2(Q)-xyz1(M) /// square of xyz2(Q)-xyz1(M) )
        pi_xyz_diff_concat = torch.cat([pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff], dim=-1)    # (B N nsample_q 3+3+3+1)
        pi_xyz_diff_concat_aft_mask = pi_xyz_diff_concat ##### mask 
        
        # Concatenate ( f2(Q) /// f1(M) /// Euclidean Space )
        pi_feat_diff = torch.cat([pi_points_expanded, qi_points_grouped],dim=-1)     # (B N nsample_q 2C)
        pi_feat1_concat = torch.cat([pi_xyz_diff_concat, pi_feat_diff], dim=-1)     # (B N nsample_q 2C+10)
        pi_feat1_concat_aft_mask = pi_feat1_concat 

        pi_feat1_new_reshape = torch.reshape(pi_feat1_concat_aft_mask, [B, H*W, self.nsample_q, -1])        # (B N nsample_q 2C+10)

        pi_xyz_diff_concat_reshape = torch.reshape(pi_xyz_diff_concat_aft_mask, [B, H*W, self.nsample_q, -1])       # (B N nsample_q 10)

        # First Flow Embedding h(Q) --- MLP(f2//f1//Eu)
        for i,conv in enumerate(self.mlp1_convs):
            pi_feat1_new_reshape = conv(pi_feat1_new_reshape)    

        # Point Position Encoding --- FC(Eu) 
        pi_xyz_encoding = self.pi_encoding(pi_xyz_diff_concat_reshape)

        # Concatenate ( FC(Eu)// h(Q) )
        pi_concat = torch.cat([pi_xyz_encoding, pi_feat1_new_reshape], dim = 3)

        # First Attentive Weight w(Q) --- MLP( FC(Eu)// h(Q) )
        for j,conv in enumerate(self.mlp2_convs):
            pi_concat = conv(pi_concat)

        valid_mask_bool = torch.eq(valid_mask, torch.ones_like(valid_mask).cuda())
        WQ_mask = valid_mask_bool.expand(B, H*W, self.nsample_q, pi_concat.shape[-1])                  #  B N K MLP[-1]     
        pi_concat_mask = torch.where(WQ_mask, pi_concat, torch.ones_like(pi_concat).cuda() * (-1e10)) 
        WQ = F.softmax(pi_concat_mask, dim=2)

        # First Attentive Flow Embedding e(M) ---  w(Q) * h(Q)
        pi_feat1_new_reshape = WQ * pi_feat1_new_reshape
        pi_feat1_new_reshape_bnc = torch.sum(pi_feat1_new_reshape, dim=2, keepdim=False)    # b, n, mlp1[-1]


        pi_feat1_new = torch.reshape(pi_feat1_new_reshape_bnc, [B, H, W, -1])  # project the selected central points to bhwc

    ##########################################################################################################################
        
        kernel_total_p = self.kernel_size1[0] * self.kernel_size1[1]

        random_HW_p = (torch.arange(0,kernel_total_p)).cuda().int()
        # nsample = kernel_size1[0] * kernel_size1[1]

        select_b_idx = torch.zeros(B, H*W, self.nsample, 1).cuda().long().detach()           # (B N nsample 1)
        select_h_idx = torch.zeros(B, H*W, self.nsample, 1).cuda().long().detach()
        select_w_idx = torch.zeros(B, H*W, self.nsample, 1).cuda().long().detach()

        valid_idx = torch.zeros(B, H*W, kernel_total_p, 1).cuda().float().detach()
        valid_in_dis_idx = torch.zeros(B, H*W, kernel_total_p, 1).cuda().float().detach()
        select_mask = torch.zeros(B, H*W, self.nsample, 1).cuda().float().detach()
        
        with torch.no_grad():
        # # Sample QNN of (M neighbour points from sampled n points in PC1) in PC1
            select_b_idx, select_h_idx, select_w_idx, valid_idx2, valid_in_dis_idx2, valid_mask2 = \
                fused_conv_random_k( warped_xyz1_proj, warped_xyz1_proj, idx_hw, random_HW_p, H, W, H * W, self.kernel_size1[0], self.kernel_size1[1], 
                                        self.nsample,  0, self.distance1, 1, 1,select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, select_mask,H,W
                )   
        
        C = pi_feat1_new.shape[3]

        neighbor_idx = select_h_idx * W + select_w_idx    
        neighbor_idx = neighbor_idx.reshape(B, -1)
        warped_xyz_bn3 = warped_xyz1_proj.reshape(B, -1, 3)
        pi_points_bn3 = pi_feat1_new.reshape(B, -1, C)
        
        pc_xyz_grouped = torch.gather(warped_xyz_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, 3))
        pc_points_grouped = torch.gather(pi_points_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, C))
        
        # Output 3D coordinates xyz1(M) of M neighbours of sampled n points in e(M) 
        # pc_points_grouped = pi_feat1_new[pc_xyz_points_b_idx, pc_xyz_points_h_idx, pc_xyz_points_w_idx, :]      # (B N nsample 1 C)
        pc_points_grouped = pc_points_grouped.reshape(B, H*W, self.nsample, C)                                   # (B N nsample C)
        pc_points_grouped= pc_points_grouped * valid_mask2

        # Output features f1(M) of M neighbours of sampled n points in PC1
        # pc_xyz_grouped = warped_xyz1_proj[pc_xyz_points_b_idx, pc_xyz_points_h_idx, pc_xyz_points_w_idx, :]     # (B N nsample 1 3)
        pc_xyz_grouped = pc_xyz_grouped.reshape(B, H*W, self.nsample, 3)                                         # (B N nsample 3)
        pc_xyz_grouped= pc_xyz_grouped * valid_mask2

        pc_xyz_new = torch.unsqueeze(warped_xyz1, dim = 2).expand(B,H*W,self.nsample,3)             # (B N nsample 3)
        pc_points_new = torch.unsqueeze(points1, dim = 2).expand(B,H*W,self.nsample,points1.shape[-1])              # (B N nsample C)

        # 3D Euclidean space information Eu ( xyz1 // xyz1(M) // xyz1-xyz1(M) // square of xyz1-xyz1(M) )
        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new                                                               # (B N nsample 3
        pc_euc_diff = torch.sqrt(torch.sum(torch.square(pc_xyz_diff), dim=-1, keepdim=True) + 1e-20)          # (B N nsample 1)
        pc_xyz_diff_concat = torch.cat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff], dim=-1)         # (B N nsample 3+3+3+1)

        # Point Position Encoding --- FC(Eu)
        pc_xyz_encoding = self.pc_encoding(pc_xyz_diff_concat)
    
        pc_concat = torch.cat([pc_xyz_encoding, pc_points_new, pc_points_grouped], dim = -1)
        pc_concat = pc_concat * valid_mask2
        
        # Second Attentive Weight w(M) --- MLP( FC(Eu)// f1(M) // xyz1(M) )
        for j,conv in enumerate(self.mlp2_convs_new):
            pc_concat = conv(pc_concat)

        valid_mask2_bool = torch.eq(valid_mask2, torch.ones_like(valid_mask2).cuda()) 
        WP_mask = valid_mask2_bool.expand(B, H*W, self.nsample, pc_concat.shape[-1])   ####   B N K MLP[-1]    #####################  
        pc_concat_mask = torch.where(WP_mask, pc_concat, torch.ones_like(pc_concat).cuda() * (-1e10)) 
        WP = F.softmax(pc_concat_mask,dim=2)   #####  b, npoints, nsample, mlp[-1]

        # Final Attentive Flow Embedding --- w(M) * xyz1(M) 
        pc_feat1_new = WP * pc_points_grouped
        pc_feat1_new = torch.sum(pc_feat1_new, dim=2, keepdim=False)           # (b n self.mlp2[-1])


        return  pc_feat1_new 


class set_upconv_module(nn.Module):
    def __init__(self, batch_size, kernel_size, H, W, stride_H, stride_W, nsample, distance, in_channels, mlp, mlp2, \
        is_training, bn_decay=None, bn=True, pooling='max', radius=None, knn=True):
        super(set_upconv_module,self).__init__()

        """
        Args:
            xyz1_proj ([type]): [description]
            xyz2_proj ([type]): [description]
            feat2_proj ([type]): [description]
            xyz1 ([type]): [frame 1 with dense points BN3]
            points1 ([type]): [frame 1 with dense points BNC]
            kernel_size ([type]): [description]
            nsample ([type]): [description]
            mlp ([type]): [description]
            mlp2 ([type]): [description]
            is_training (bool): [description]
            bn_decay ([type], optional): [description]. Defaults to None.
            bn (bool, optional): [description]. Defaults to True.
            pooling (str, optional): [description]. Defaults to 'max'.
            radius ([type], optional): [description]. Defaults to None.
            knn (bool, optional): [description]. Defaults to True.
        Returns:
            BNC feature
        """

        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.nsample = nsample
        self.mlp = mlp
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.radius = radius
        self.knn = knn
        self.stride_H = stride_H
        self.stride_W = stride_W
        self.distance = distance

        self.last_channel = in_channels[-1] + 3  # LAST CHANNEL = C+3
        self.mlp_conv = nn.ModuleList()
        self.mlp2_conv = nn.ModuleList()

        self.idx_n2 = get_hw_idx(batch_size, H, W, 1, 1)


        if mlp is not None:
            for i,num_out_channel in enumerate(mlp):
                self.mlp_conv.append(Conv2d(self.last_channel,num_out_channel,[1,1],stride=[1,1], bn=True ))
                self.last_channel = num_out_channel

        if len(mlp) is not 0:
            self.last_channel = mlp[-1] + in_channels[0]
        else:
            self.last_channel = self.last_channel + in_channels[0]

        if mlp2 is not None:
            for i,num_out_channel in enumerate(mlp2):
                self.mlp2_conv.append(Conv2d(self.last_channel,num_out_channel,[1,1],stride=[1,1], bn=True))
                self.last_channel = num_out_channel

        #  xyz1 dense points         #  xyz2 sparse points (queried)

    def forward(self, xyz1_proj, xyz2_proj, points1_proj, feat2_proj):
        
        B = xyz1_proj.shape[0]
        H = xyz1_proj.shape[1]
        W = xyz1_proj.shape[2]
        C = feat2_proj.shape[3]

        SMALL_H = xyz2_proj.shape[1]
        SMALL_W = xyz2_proj.shape[2]

        xyz1 = xyz1_proj.reshape(B, -1, 3)
        points1 = points1_proj.reshape(B, -1, points1_proj.shape[-1])

        idx_hw = self.idx_n2.cuda().int().contiguous()                               ###  B N 2
        
        kernel_total = self.kernel_size[0] * self.kernel_size[1]
        random_HW = (torch.arange(kernel_total)).cuda().int()
   
        select_b_idx = torch.zeros(B, H*W, self.nsample, 1).cuda().long().detach()       # B N n_sample 1
        select_h_idx = torch.zeros(B, H*W, self.nsample, 1).cuda().long().detach()
        select_w_idx = torch.zeros(B, H*W, self.nsample, 1).cuda().long().detach()

        valid_idx = torch.zeros(B, H*W, kernel_total, 1).cuda().float().detach()         # B N kernel_total 1
        valid_in_dis_idx = torch.zeros(B, H*W, kernel_total, 1).cuda().float().detach()
        select_mask = torch.zeros(B, H*W, self.nsample, 1).cuda().float().detach()       # B N n_sample 1


        with torch.no_grad():

        # output the KNN of n dense points in n' sparse points (skip connection) 
            xyz1_up_xyz_points_b_idx, xyz1_up_xyz_points_h_idx, xyz1_up_xyz_points_w_idx, valid_idx, valid_in_dis_idx, valid_mask = fused_conv_random_k(
                xyz1_proj, xyz2_proj, idx_hw, random_HW, H, W, H*W, 
                self.kernel_size[0], self.kernel_size[1], self.nsample, 1, self.distance,
                self.stride_H, self.stride_W, select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, select_mask,SMALL_H,SMALL_W
            )

        
        # output the xyz1(K) of KNN points of dense n points 
        xyz1_up_grouped = xyz2_proj [xyz1_up_xyz_points_b_idx, xyz1_up_xyz_points_h_idx, xyz1_up_xyz_points_w_idx, :]       # (B npoints nsample 1 3)
        xyz1_up_grouped = xyz1_up_grouped.reshape(B, H*W, self.nsample, 3)                                                   # (B npoints nsample 3)
        xyz1_up_grouped = xyz1_up_grouped * valid_mask
        #print("xyz1_up_grouped:",xyz1_up_grouped.shape)
        
        # output the feature f1(K) of KNN points of dense n points
        xyz1_up_points_grouped = feat2_proj [xyz1_up_xyz_points_b_idx, xyz1_up_xyz_points_h_idx, xyz1_up_xyz_points_w_idx, :]           # (B npoints nsample 1 in_channel[-1])           
        xyz1_up_points_grouped = xyz1_up_points_grouped.reshape(B, H*W, self.nsample, C)                                                 # (B npoints nsample in_channel[-1])
        xyz1_up_points_grouped = xyz1_up_points_grouped * valid_mask

        xyz1_expanded = torch.unsqueeze(xyz1, 2).expand(B, H*W, self.nsample, 3)      # (B, H*W, nsample, 3)
        
        #  Concatenate ( xyz1(K)-xyz1 // f1(K) )
        xyz1_diff = xyz1_up_grouped - xyz1_expanded                                 # (B, H*W, nsample, 3)
        xyz1_concat = torch.cat([xyz1_diff, xyz1_up_points_grouped], dim = -1)     # (B, H*W, nsample, in_channel[-1]+3)
        xyz1_concat_aft_mask = xyz1_concat
        xyz1_concat_aft_mask_reshape = torch.reshape(xyz1_concat_aft_mask, [B, H*W, self.nsample, -1])  # B, npoint1, nsample, in_channel[-1]+3
        
        # Point Feature Embedding -- shared MLP 
        for i,conv in enumerate(self.mlp_conv):
            xyz1_concat_aft_mask_reshape = conv(xyz1_concat_aft_mask_reshape)       # (B, H*W, nsample, mlp[-1])
     
       # Pooling in Local Regions -- max pooling (gather K points feature to 1)
        if self.pooling == 'max':
            xyz1_up_feat = torch.max(xyz1_concat_aft_mask_reshape, dim=2, keepdim=False)[0]       # (B, H*W,  mlp[-1])
        if self.pooling == 'avg':
            xyz1_up_feat = torch.mean(xyz1_concat_aft_mask_reshape, dim=2, keepdim=False)

    #############################   mlp2  ##########################

        xyz1_up_feat_concat_feat1 = torch.cat([xyz1_up_feat, points1], dim = -1)       # B  H*W  mlp[-1]+in_channel[0] 
        xyz1_up_feat_concat_feat1 = torch.unsqueeze(xyz1_up_feat_concat_feat1, 2)   # B  H*W  1 mlp[-1]+in_channel[0]  

        # Further Processing --- another MLP
        for i,conv in enumerate(self.mlp2_conv):
            xyz1_up_feat_concat_feat1 = conv(xyz1_up_feat_concat_feat1)     # B  H*W  1  mlp2[-1]

        xyz1_up_feat_concat_feat1 = torch.squeeze(xyz1_up_feat_concat_feat1, 2) # B  H*W  mlp2[-1]

        #print("xyz1_up_feat_concat_feat1:",xyz1_up_feat_concat_feat1.shape)
        #print("------------------- UpConv End ------------------")

        return xyz1_up_feat_concat_feat1




class FlowPredictor(nn.Module):

    def __init__(self, in_channels, mlp, is_training, bn_decay, bn=True):
        super(FlowPredictor, self).__init__()
        self.in_channels = in_channels
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.mlp_conv = nn.ModuleList()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i, num_out_channel in enumerate(mlp):
            self.mlp_conv.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn))
            self.in_channels = num_out_channel

    def forward(self, points_f1, upsampled_feat, cost_volume):

        '''
                    Input:
                        points_f1: (b,n,c1)
                        upsampled_feat: (b,n,c2)
                        cost_volume: (b,n,c3)

                    Output:
                        points_concat:(b,n,mlp[-1])
                '''
        if upsampled_feat is not None:
            points_concat = torch.cat([points_f1, cost_volume, upsampled_feat], -1)  # b,n,c1+c2+c3
        else:
            points_concat = torch.cat([points_f1, cost_volume], -1)

        points_concat = torch.unsqueeze(points_concat, 2)  # B,n,1,c1+c2+c3

        for i, conv in enumerate(self.mlp_conv):
            points_concat = conv(points_concat)

        points_concat = torch.squeeze(points_concat, 2)

        return points_concat

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes )
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes, 2*stride)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        return out
def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)
BatchNorm2d = nn.BatchNorm2d

