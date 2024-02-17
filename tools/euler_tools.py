# -*- coding:UTF-8 -*-

import math
import numpy as np
# author:Zhiheng Feng
# contact: fzhsjtu@foxmail.com
# datetime:2021/10/21 19:56
# software: PyCharm

"""
文件说明：变换矩阵和四元数之间的转换
    
"""


def mat2euler(M, cy_thresh=None, seq='zyx'):
    M = np.asarray(M)
    if cy_thresh is None:
        cy_thresh = np.finfo(M.dtype).eps * 4

    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33 * r33 + r23 * r23)
    if seq == 'zyx':
        if cy > cy_thresh:  # cos(y) not close to zero, standard form
            z = math.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21, r22)
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = 0.0
    elif seq == 'xyz':
        if cy > cy_thresh:
            y = math.atan2(-r31, cy)
            x = math.atan2(r32, r33)
            z = math.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi / 2
                x = math.atan2(r12, r13)
            else:
                y = -np.pi / 2
    else:
        raise Exception('Sequence not recognized')
    return z, y, x

def euler2quat(z=0, y=0, x=0, isRadian=True):
    ''' Return quaternion corresponding to these Euler angles
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
        Rotation angle in radians around z-axis (performed first)
    y : scalar
        Rotation angle in radians around y-axis
    x : scalar
        Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    quat : array shape (4,)
        Quaternion in w, x, y z (real, then vector) format
    Notes
    -----
    We can derive this formula in Sympy using:
    1. Formula giving quaternion corresponding to rotation of theta radians
        about arbitrary axis:
        http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
        theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
        http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
        formulae from 2.) to give formula for combined rotations.
    '''

    if not isRadian:
        z = ((np.pi) / 180.) * z
        y = ((np.pi) / 180.) * y
        x = ((np.pi) / 180.) * x
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
        cx * cy * cz - sx * sy * sz,
        cx * sy * sz + cy * cz * sx,
        cx * cz * sy - sx * cy * sz,
        cx * cy * sz + sx * cz * sy])

def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion'''
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X;
    wY = w * Y;
    wZ = w * Z
    xX = x * X;
    xY = x * Y;
    xZ = x * Z
    yY = y * Y;
    yZ = y * Z;
    zZ = z * Z
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])

    return 0
