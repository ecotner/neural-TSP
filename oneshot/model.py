import torch
import torch.nn as nn
from torch_cluster import graclus_cluster


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
