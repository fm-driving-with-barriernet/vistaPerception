import math
import itertools
from functools import partial
import sys

import torch
import torch.nn.functional as F
import numpy as np

from dinov2.dinov2.eval.depth.models import build_depther

"""
Purpose of this file:
This file will handle running inference for depth estimation on an image fed from VISTA display.render()
It will return the depth estimation png result to later be used in downstream inputs to BarrierNet
"""

def run_depth_inference():
    """
    stuff here
    """
    return None