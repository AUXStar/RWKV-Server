import torch

torch.cuda.init()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from .loader import RWKV070ModelLoader
from .manager_simple import SimpleScheduler
from .manager_dynamic import DynamicScheduler
