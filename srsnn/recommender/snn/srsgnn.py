import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class SRSGNN(nn.Module):
    def __init__(self, item_num, params):
        super(SRSGNN, self).__init__()