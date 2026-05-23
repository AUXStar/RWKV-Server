import torch
import gc, os, re
from .rwkv.faster3a_2605 import rwkv7_fast_v3a as v3a

class RWKV7(v3a.RWKV7):
    def __init__(self, model_path):
        v3a.MODEL_PATH = model_path
        v3a.WKV_MODE = "fp16"
        v3a.EMB_DEVICE = "cpu"
        v3a.RKV_MODE = "off"
        v3a.CMIX_SPARSE = "no-fc"
        v3a.LOWRANK_WEIGHT = "transpose"
        v3a.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
        v3a.load_extensions(v3a.WKV_MODE)
        super().__init__()
    