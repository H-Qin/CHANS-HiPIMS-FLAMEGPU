import sys
import os
from numpy.lib.scimath import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import time
import math
import randomTest

deviceID = 0

torch.cuda.set_device(deviceID)

device = torch.device("cuda", deviceID)

h = torch.zeros(10, device=device)
randomTest.update(h)
print(h)