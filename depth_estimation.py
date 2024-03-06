import math
import itertools
from functools import partial
import sys

import torch
import torch.nn.functional as F

from dinov2.dinov2.eval.depth.models import build_depther