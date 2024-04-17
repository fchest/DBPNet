from __future__ import print_function
import numpy as np
import math
np.set_printoptions(suppress=True)
import os
import time
import torch


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x**2 + y**2
    r = math.sqrt(x2_y2 + z**2)                    # r
    elev = math.atan2(z, math.sqrt(x2_y2))            # Elevation
    az = math.atan2(y, x)                          # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * math.cos(theta), rho * math.sin(theta)


def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def monitor(process, multiple, second):
    while True:
        sum = 0
        for ps in process:
            if ps.is_alive():
                sum += 1
        if sum < multiple:
            break
        else:
            time.sleep(second)

def save_load_name(args, name=''):
    name = name if len(name) > 0 else 'default_model'
    return name

def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'./pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'./pre_trained_models/{name}.pt')
    return model