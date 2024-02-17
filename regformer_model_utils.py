import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import os
from torch import Tensor

_EPS = 1e-6



def ProjectPCimg2SphericalRing(PC, Feature = None, H_input = 64, W_input = 1800):

    batch_size = len(PC)

    if Feature != None:
        num_channel = Feature[0].shape[-1]

    degree2radian = math.pi / 180
    nLines = H_input    
    AzimuthResolution = 360.0 / W_input # degree
    VerticalViewDown = -24.8
    VerticalViewUp = 2.0

    # specifications of Velodyne-64
    AzimuthResolution = AzimuthResolution * degree2radian
    VerticalViewDown = VerticalViewDown * degree2radian
    VerticalViewUp = VerticalViewUp * degree2radian
    VerticalResolution = (VerticalViewUp - VerticalViewDown) / (nLines - 1)
    VerticalPixelsOffset = -VerticalViewDown / VerticalResolution

    # parameters for spherical ring's bounds
    
    PI = math.pi

    for batch_idx in range(batch_size):

        ###  initialize current processed frame 
            
        cur_PC = PC[batch_idx].to(torch.float32)  # N  3
        # print(cur_PC.shape)
        if Feature != None:
            cur_Feature = Feature[batch_idx]  # N  c

        x = cur_PC[:, 0] 
        y = cur_PC[:, 1] 
        z = cur_PC[:, 2]

        r = torch.norm(cur_PC, p=2, dim =1)

        PC_project_current = torch.zeros([H_input, W_input, 3]).cuda().detach()  # shape H W 3
        if Feature != None:
            Feature_project_current = torch.zeros([H_input, W_input, num_channel]).cuda().detach()

        ####  get iCol & iRow
                
        iCol = ((PI -torch.atan2(y, x))/ AzimuthResolution) # alpha
        iCol = iCol.to(torch.int32)

        beta = torch.asin(z/r)                              # beta

        tmp_int = (beta / VerticalResolution + VerticalPixelsOffset)
        tmp_int = tmp_int.to(torch.int32)

        iRow = H_input - tmp_int

        iRow = torch.clamp(iRow, 0, H_input - 1)
        iCol = torch.clamp(iCol, 0, W_input - 1)

        iRow = iRow.to(torch.long)  # N 1
        iCol = iCol.to(torch.long)  # N 1



        PC_project_current[iRow, iCol, :] = cur_PC[:, :]  # H W 3
        if Feature != None:
            Feature_project_current[iRow, iCol, :] = cur_Feature[:, :]


        # Generate mask

        PC_mask_valid = torch.any(PC_project_current != 0, dim=-1).cuda().detach()  # H W
        PC_mask_valid = torch.unsqueeze(PC_mask_valid, dim=2).to(torch.float32) # H W 1

        if Feature != None:
            Feature_mask_valid = ~torch.any(Feature_project_current!= 0, dim=-1).cuda().detach()  # H W
            Feature_mask_valid = torch.unsqueeze(Feature_mask_valid, dim=2).to(torch.float32)

        ####1 h w
        PC_project_current = torch.unsqueeze(PC_project_current, dim=0)
        PC_mask_valid = torch.unsqueeze(PC_mask_valid, dim=0)

        if Feature != None:
            Feature_project_current = torch.unsqueeze(Feature_project_current,dim=0)
            Feature_mask_valid = torch.unsqueeze(Feature_mask_valid, dim=0)
        ####b h w
        if batch_idx == 0:
            PC_project_final = PC_project_current
            PC_mask_final = PC_mask_valid
            if Feature != None:
                Feature_project_final = Feature_project_current
                Feature_mask_final = Feature_mask_valid

        else:
            PC_project_final = torch.cat([PC_project_final, PC_project_current], 0)  # b h w 3
            PC_mask_final = torch.cat([PC_mask_final, PC_mask_valid], 0)
            if Feature != None:
                Feature_project_final = torch.cat([Feature_project_final, Feature_project_current], 0)
                Feature_mask_final = torch.cat([Feature_mask_final, Feature_mask_valid], 0)


    if Feature != None:
        return PC_project_final,  Feature_project_final
    else:
        return PC_project_final,  PC_mask_final



def quatt2T(q, t):
    
    ''' Calculate rotation matrix corresponding to quaternion
    Parameters
    ----------
    q : 4 element array-like
    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*
    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    '''
    t0 = t[0]; t1 = t[1]; t2 = t[2]
    w = q[0]; x = q[1]; y = q[2]; z = q[3]
    Nq = w*w + x*x + y*y + z*z
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z

    c1 = torch.as_tensor(1.0).cuda()
    add = torch.as_tensor([[1.0, 0, 0, 0]]).cuda()

    T = ([[ c1-(yY+zZ), xY-wZ, xZ+wY, t0],
            [ xY+wZ, c1-(xX+zZ), yZ-wX, t1],
            [ xZ-wY, yZ+wX, c1-(xX+yY), t2]])

    T = torch.cat([T, add], 0)

    return  T

def euler2quat(z, y, x):

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    return torch.tensor([cx*cy*cz - sx*sy*sz,
                    cx*sy*sz + cy*cz*sx,
                    cx*cz*sy - sx*cy*sz,
                    cx*cy*sz + sx*cz*sy]).cuda()

def mat2euler(M, seq='zyx'):

    r11 = M[0, 0]; r12 = M[0, 1]; r13 = M[0, 2]
    r21 = M[1, 0]; r22 = M[1, 1]; r23 = M[1, 2]
    r31 = M[2, 0]; r32 = M[2, 1]; r33 = M[2, 2]

    cy = torch.sqrt(r33*r33 + r23*r23)

    z = torch.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
    y = torch.atan2(r13,  cy) # atan2(sin(y), cy)
    x = torch.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))

    return z, y, x

def AugQt(q_input, t_input, T_all, T_all_inv):

    batch_size = q_input.shape[0]

    for i in range(batch_size):
            
        cur_q_input = torch.reshape(q_input[i, :, :], [4])
        cur_t_input = torch.reshape(t_input[i, :, :], [3])

        cur_T_all = T_all[i, :, :]
        cur_T_all_inv = T_all_inv[i, :, :]

        cur_T0 = quatt2T(cur_q_input, cur_t_input)

        cur_T_out = torch.mm(cur_T_all_inv, cur_T0)
        cur_T_out = torch.mm(cur_T_out, cur_T_all)

        cur_R_out = cur_T_out[:3, :3]  ###  3 3
        cur_t_out = torch.reshape(cur_T_out[:3, 3:], [1, 1, 3])  ###  1 1 3

        z_euler, y_euler, x_euler = mat2euler(cur_R_out)
        cur_q_out = torch.reshape(euler2quat(z_euler, y_euler, x_euler), [1, 1, 4])  ####  1 1 4
        
        if i == 0:
            q_out = cur_q_out
            t_out = cur_t_out

        else:
            q_out = torch.cat([q_out, cur_q_out], 0)    # b h w 3   
            t_out = torch.cat([t_out, cur_t_out], 0)    # b h w 3   

    return q_out, t_out

def softmax_valid(feature_bnc, weight_bnc, mask_valid):

    batch_size = feature_bnc.shape[0]

    for b in range(batch_size):

        feature_bnc_current = feature_bnc[b, :, :]  ##   N C
        weight_bnc_current = weight_bnc[b, :, :]  ##   N C
        mask_valid_current = mask_valid[b, :]  ## N'

        feature_bnc_current_valid = feature_bnc_current[mask_valid_current > 0, :]  ## N' C
        weight_bnc_current_valid = weight_bnc_current[mask_valid_current > 0, :]  ###  N' C

        W_softmax = F.softmax(weight_bnc_current_valid, dim=0)
        feature_new_current = torch.sum(feature_bnc_current_valid * W_softmax, dim = 0, keepdim = True)

        feature_new_current = torch.reshape(feature_new_current, [1, 1, -1])

        if b == 0:
            feature_new_final = feature_new_current
        else:
            feature_new_final = torch.cat([feature_new_final, feature_new_current], 0)     #    B 1 C 

    return feature_new_final


def PreProcess(PC_f1, PC_f2, T_gt):    ####    pre process procedure

    batch_size = len(PC_f1)
    PC_f1_concat = []
    PC_f2_concat = []
    PC_f1_aft_aug = []
    PC_f2_aft_aug = []



    for p in PC_f1:
        num_points = torch.tensor(p.shape[0], dtype=torch.int32)
        add_T = torch.ones(num_points, 1).cuda().to(torch.float32)
        PC_f1_add = torch.cat([p, add_T], -1)  ##  concat one to form   n 4
        PC_f1_concat.append(PC_f1_add)

    for p in PC_f2:
        num_points = torch.tensor(p.shape[0], dtype=torch.int32)
        add_T = torch.ones(num_points, 1).cuda().to(torch.float32)
        PC_f2_add = torch.cat([p, add_T], -1)  ##  concat one to form   n 4
        PC_f2_concat.append(PC_f2_add)
    # print([m.shape for m in PC_f1_concat])


    for i in range(batch_size):

        cur_T_gt = T_gt[i, :, :].to(torch.float32)

        cur_PC_f1_concat = PC_f1_concat[i]
        cur_PC_f2_concat = PC_f2_concat[i]


        ##  select the 50m * 50m region ########
        r_f1 = torch.norm(cur_PC_f1_concat[:, :2], p=2, dim =1, keepdim = True).repeat(1, 4)
        cur_PC_f1_concat = torch.where( r_f1 > 50 , torch.zeros_like(cur_PC_f1_concat).cuda(), cur_PC_f1_concat ).to(torch.float32)
        PC_mask_valid1 = torch.any(cur_PC_f1_concat != 0, dim=-1).cuda().detach()  # H W
        cur_PC_f1_concat = cur_PC_f1_concat[PC_mask_valid1 > 0,:]

        r_f2 = torch.norm(cur_PC_f2_concat[:, :2], p=2, dim =1, keepdim = True).repeat(1, 4)
        cur_PC_f2_concat = torch.where( r_f2 > 50 , torch.zeros_like(cur_PC_f2_concat).cuda(), cur_PC_f2_concat ).to(torch.float32)
        PC_mask_valid2 = torch.any(cur_PC_f2_concat != 0, dim=-1).cuda().detach()  # H W
        cur_PC_f2_concat = cur_PC_f2_concat[PC_mask_valid2 > 0, :]


        #####   generate  the  valid  mask (remove the not valid points)
        mask_valid_f1 = torch.any(cur_PC_f1_concat != 0, dim=-1, keepdim=True).cuda().detach()  # N 1
        mask_valid_f1 = mask_valid_f1.to(torch.float32)

        mask_valid_f2 = torch.any(cur_PC_f2_concat != 0, dim=-1, keepdim=True).cuda().detach()  # N 1
        mask_valid_f2 = mask_valid_f2.to(torch.float32)


        cur_PC_f1_concat = cur_PC_f1_concat[:, :3]
        cur_PC_f2_concat = cur_PC_f2_concat[:, :3]

        cur_PC_f1_mask = cur_PC_f1_concat * mask_valid_f1 # N 3
        cur_PC_f2_mask = cur_PC_f2_concat * mask_valid_f2

        cur_R_gt = cur_T_gt[:3, :3]  ###  3 3
        z_euler, y_euler, x_euler = mat2euler(cur_R_gt)
        cur_q_gt = euler2quat(z_euler, y_euler, x_euler)  ####  4
        cur_q_gt = torch.unsqueeze(cur_q_gt, dim=0)
        cur_t_gt = cur_T_gt[:3, 3:]  ###   3 1
        cur_t_gt = torch.unsqueeze(cur_t_gt, dim=0)

        if i ==0:
            q_gt = cur_q_gt
            t_gt = cur_t_gt
        else:
            q_gt = torch.cat([q_gt, cur_q_gt], 0)  ##array
            t_gt = torch.cat([t_gt, cur_t_gt], 0)

        PC_f1_aft_aug.append(cur_PC_f1_mask)### list
        PC_f2_aft_aug.append(cur_PC_f2_mask)



    return PC_f1_aft_aug, PC_f2_aft_aug, q_gt, t_gt


def quat2mat(q):
    '''
    :param q: Bx4
    :return: R: BX3X3
    '''
    batch_size = q.shape[0]
    w, x, y, z = q[:, 0].unsqueeze(1), q[:, 1].unsqueeze(1), q[:, 2].unsqueeze(1), q[:, 3].unsqueeze(1)
    Nq = torch.sum(q ** 2, dim=1, keepdim=True)
    s = 2.0 / Nq
    wX = w * x * s; wY = w * y * s; wZ = w * z * s
    xX = x * x * s; xY = x * y * s; xZ = x * z * s
    yY = y * y * s; yZ = y * z * s; zZ = z * z * s
    a1 = 1.0 - (yY + zZ); a2 = xY - wZ; a3 = xZ + wY
    a4 = xY + wZ; a5 = 1.0 - (xX + zZ); a6 = yZ - wX
    a7 = xZ - wY; a8 = yZ + wX; a9 = 1.0 - (xX + yY)
    R = torch.cat([a1, a2, a3, a4, a5, a6, a7, a8, a9], dim=1).view(batch_size, 3, 3)
    return R


def inv_q(q, batch_size):

    q = torch.squeeze(q, dim=1)
    q_2 = torch.sum(q * q, dim=-1, keepdim=True) + 1e-10
    q0 = torch.index_select(q, 1, torch.LongTensor([0]).cuda())
    q_ijk = -torch.index_select(q, 1, torch.LongTensor([1, 2, 3]).cuda())
    q_ = torch.cat([q0, q_ijk], dim=-1)
    q_inv = q_ / q_2

    return q_inv


def mul_q_point(q_a, q_b, batch_size):
    q_a = torch.reshape(q_a, [batch_size, 1, 4])

    q_result_0 = torch.mul(q_a[:, :, 0], q_b[:, :, 0]) - torch.mul(q_a[:, :, 1], q_b[:, :, 1]) - torch.mul(
        q_a[:, :, 2], q_b[:, :, 2]) - torch.mul(q_a[:, :, 3], q_b[:, :, 3])
    q_result_0 = torch.reshape(q_result_0, [batch_size, -1, 1])

    q_result_1 = torch.mul(q_a[:, :, 0], q_b[:, :, 1]) + torch.mul(q_a[:, :, 1], q_b[:, :, 0]) + torch.mul(
        q_a[:, :, 2], q_b[:, :, 3]) - torch.mul(q_a[:, :, 3], q_b[:, :, 2])
    q_result_1 = torch.reshape(q_result_1, [batch_size, -1, 1])

    q_result_2 = torch.mul(q_a[:, :, 0], q_b[:, :, 2]) - torch.mul(q_a[:, :, 1], q_b[:, :, 3]) + torch.mul(
        q_a[:, :, 2], q_b[:, :, 0]) + torch.mul(q_a[:, :, 3], q_b[:, :, 1])
    q_result_2 = torch.reshape(q_result_2, [batch_size, -1, 1])

    q_result_3 = torch.mul(q_a[:, :, 0], q_b[:, :, 3]) + torch.mul(q_a[:, :, 1], q_b[:, :, 2]) - torch.mul(
        q_a[:, :, 2], q_b[:, :, 1]) + torch.mul(q_a[:, :, 3], q_b[:, :, 0])
    q_result_3 = torch.reshape(q_result_3, [batch_size, -1, 1])

    q_result = torch.cat([q_result_0, q_result_1, q_result_2, q_result_3], dim=-1)

    return q_result  ##  B N 4


def mul_point_q(q_a, q_b, batch_size):
    q_b = torch.reshape(q_b, [batch_size, 1, 4])

    q_result_0 = torch.mul(q_a[:, :, 0], q_b[:, :, 0]) - torch.mul(q_a[:, :, 1], q_b[:, :, 1]) - torch.mul(
        q_a[:, :, 2], q_b[:, :, 2]) - torch.mul(q_a[:, :, 3], q_b[:, :, 3])
    q_result_0 = torch.reshape(q_result_0, [batch_size, -1, 1])

    q_result_1 = torch.mul(q_a[:, :, 0], q_b[:, :, 1]) + torch.mul(q_a[:, :, 1], q_b[:, :, 0]) + torch.mul(
        q_a[:, :, 2], q_b[:, :, 3]) - torch.mul(q_a[:, :, 3], q_b[:, :, 2])
    q_result_1 = torch.reshape(q_result_1, [batch_size, -1, 1])

    q_result_2 = torch.mul(q_a[:, :, 0], q_b[:, :, 2]) - torch.mul(q_a[:, :, 1], q_b[:, :, 3]) + torch.mul(
        q_a[:, :, 2], q_b[:, :, 0]) + torch.mul(q_a[:, :, 3], q_b[:, :, 1])
    q_result_2 = torch.reshape(q_result_2, [batch_size, -1, 1])

    q_result_3 = torch.mul(q_a[:, :, 0], q_b[:, :, 3]) + torch.mul(q_a[:, :, 1], q_b[:, :, 2]) - torch.mul(
        q_a[:, :, 2], q_b[:, :, 1]) + torch.mul(q_a[:, :, 3], q_b[:, :, 0])
    q_result_3 = torch.reshape(q_result_3, [batch_size, -1, 1])

    q_result = torch.cat([q_result_0, q_result_1, q_result_2, q_result_3], dim=-1)

    return q_result  ##  B N 4




if __name__=='__main__':

    BATCH_SIZE = 1

    torch.cuda.set_device(1)

    T_eye = np.eye(4)
    T_eye = np.reshape(T_eye, [1, 4, 4])


    seq = "00"
    frame = 5



    file_path1 = "/tmp/kitti/odometry/data_odometry_velodyne/dataset/sequences/" + seq + "/velodyne/" + str(frame).zfill(6) + ".bin"
    file_path2 = "/tmp/kitti/odometry/data_odometry_velodyne/dataset/sequences/" + seq + "/velodyne/" + str(frame+1).zfill(6) + ".bin"

    point1 = np.fromfile(file_path1, dtype=np.float32).reshape(-1, 4)
    point2 = np.fromfile(file_path2, dtype=np.float32).reshape(-1, 4)       

    n1 = point1.shape[0]
    n2 = point2.shape[0]
    
    pos1 = np.zeros((1, 150000, 3), dtype=float)
    pos2 = np.zeros((1, 150000, 3), dtype=float)

    pos1[:, :n1, :] = point1[:, :3]
    pos2[:, :n2, :] = point2[:, :3]

    batch_size = 1

    pos1 = torch.from_numpy(pos1)
    pos2 = torch.from_numpy(pos2)

    pos1 = pos2.cuda().to(torch.float32)
    pos2 = pos1.cuda().to(torch.float32)

    T_gt = torch.eye(4).unsqueeze(0).cuda()
    T_trans = torch.eye(4).unsqueeze(0).cuda()
    T_trans_inv = torch.eye(4).unsqueeze(0).cuda()

    start = time.time()

    for i in range(100):
        xyz_f1_aug, xyz_f2_aug, q_gt, t_gt = PreProcess(pos1, pos2, T_gt)
        pc1_proj = ProjectPCimg2SphericalRing(pos1, Feature = None, H_input = 64, W_input = 1800)
        pc1_proj = ProjectPCimg2SphericalRing(pos2, Feature = None, H_input = 64, W_input = 1800)


    print('proj_total_time: ', time.time() - start)

