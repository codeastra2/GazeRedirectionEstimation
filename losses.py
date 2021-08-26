# FIle usesd for computing Gaze Estimation angular loss.
# Source: https://github.com/zhengyuf/STED-gaze/blob/master/losses.py

import torch
import torch.nn.functional as F
import numpy as np

def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, eps=1e-6)
    sim = F.hardtanh(sim, -1.0, 1.0)
    return torch.acos(sim) * (180 / np.pi)


def pitchyaw_to_vector(pitchyaws):
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)


def gaze_angular_loss(y, y_hat):
    y = pitchyaw_to_vector(y)
    y_hat = pitchyaw_to_vector(y_hat)
    loss = nn_angular_distance(y, y_hat)

    # Here we do not compute the mean as it would be better to have 
    # all the values and do the required manipulation at a later stage.  
    return loss
