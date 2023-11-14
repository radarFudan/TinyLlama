import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from flash_attn import flash_attn_func
from lit_gpt.config import Config
from xformers.ops import SwiGLU
from lit_gpt.fused_rotary_embedding import apply_rotary_emb_func
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

from lit_gpt.associative_scan import associative_scan, nested_func

from lit_gpt.retnet import RetNetDecoder
from lit_gpt.retnet_config import RetNetConfig

import einops


retnet_config = RetNetConfig()
args = {}
retnet_config.override(args) # Can you a config file to overwrite the previous config. 
attn = RetNetDecoder(retnet_config)

inputs = torch.randn(1, 1024, 768).to(device="cuda")
print("inputs device is", inputs.device)
attn.to(device="cuda")
outputs, _ = attn(inputs, None, None)

print("outputs device is", outputs.shape)

print("passed")

