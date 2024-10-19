import math

import torch
import loralib as lora
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class LoraMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, torch_tgt_module, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, lora_r=1, lora_alpha=0.):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        self.lora_r = lora_r
        self.merged = False
        self.merge_weights = True
        self.lora_alpha = lora_alpha

        self.lora_Q_A = nn.Parameter(self.in_proj_weight.new_zeros(lora_r, embed_dim))
        self.lora_Q_B = nn.Parameter(self.in_proj_weight.new_zeros(embed_dim, lora_r))
        self.lora_K_A = nn.Parameter(self.in_proj_weight.new_zeros(lora_r, embed_dim))
        self.lora_K_B = nn.Parameter(self.in_proj_weight.new_zeros(embed_dim, lora_r))
        self.lora_V_A = nn.Parameter(self.in_proj_weight.new_zeros(lora_r, embed_dim))
        self.lora_V_B = nn.Parameter(self.in_proj_weight.new_zeros(embed_dim, lora_r))
        self.lora_out_proj_A = nn.Parameter(self.out_proj.weight.new_zeros(lora_r, embed_dim))
        self.lora_out_proj_B = nn.Parameter(self.out_proj.weight.new_zeros(embed_dim, lora_r))
        self.scaling = self.lora_alpha / self.lora_r
        self.set_parameters(torch_tgt_module)

    def set_parameters(self, torch_tgt_module):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_Q_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_Q_B)
        nn.init.kaiming_uniform_(self.lora_K_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_K_B)
        nn.init.kaiming_uniform_(self.lora_V_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_V_B)
        nn.init.kaiming_uniform_(self.lora_out_proj_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_out_proj_B)
        assert isinstance(torch_tgt_module, nn.MultiheadAttention)        
        assert self.embed_dim == torch_tgt_module.embed_dim
        assert self.batch_first == torch_tgt_module.batch_first
        assert self.dropout == torch_tgt_module.dropout
        assert self.head_dim == torch_tgt_module.head_dim
        assert self.num_heads == torch_tgt_module.num_heads
        assert self.kdim == torch_tgt_module.kdim
        assert self.vdim == torch_tgt_module.vdim
        self.in_proj_weight.data = torch_tgt_module.in_proj_weight.data
        self.in_proj_bias.data = torch_tgt_module.in_proj_bias.data
        self.out_proj.weight.data = torch_tgt_module.out_proj.weight.data
        self.out_proj.bias.data = torch_tgt_module.out_proj.bias.data

    def train(self, mode: bool = True):
        nn.MultiheadAttention.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.lora_r > 0:
                    deltQ = self.lora_Q_B @ self.lora_Q_A
                    deltK = self.lora_K_B @ self.lora_K_A
                    deltV = self.lora_V_B @ self.lora_V_A
                    self.in_proj_weight.data -= torch.cat((deltQ, deltK, deltV), dim=0) * self.scaling
                    self.out_proj.weight.data -= (self.lora_out_proj_B @ self.lora_out_proj_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.lora_r > 0:
                    deltQ = self.lora_Q_B @ self.lora_Q_A
                    deltK = self.lora_K_B @ self.lora_K_A
                    deltV = self.lora_V_B @ self.lora_V_A
                    self.in_proj_weight.data += torch.cat((deltQ, deltK, deltV), dim=0) * self.scaling
                    self.out_proj.weight.data += (self.lora_out_proj_B @ self.lora_out_proj_A) * self.scaling
                self.merged = True

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None):
        # Merge = False
        if self.lora_r > 0 and not self.merged:
            
            if self.lora_r > 0:
                deltQ = self.lora_Q_B @ self.lora_Q_A
                deltK = self.lora_K_B @ self.lora_K_A
                deltV = self.lora_V_B @ self.lora_V_A
                lora_in_proj_weight = self.in_proj_weight.data + torch.cat((deltQ, deltK, deltV), dim=0) * self.scaling
                lora_out_proj_weight = self.out_proj.weight.data + (self.lora_out_proj_B @ self.lora_out_proj_A) * self.scaling

                result = F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    lora_in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, lora_out_proj_weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask)
            return result
        # Merge =True
        else:
            return nn.MultiheadAttention.forward(self, query, key, value, key_padding_mask=key_padding_mask,
                                                 need_weights=need_weights, attn_mask=attn_mask)



def lora_replace_attention_layers(
    transformer: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    start_block: int = 0,
):
    """
    Utility function to replace attention layers of a CLIP transformer model
    with LoRAMultiHeadAttention. It expects a pre-defined structure (following OpenAI's CLIP).

    Args:
        transformer (nn.Module): transformer to replace attention layers.
        lora_r (int): LoRA's rank.
        lora_alpha (int): LoRA's alpha for scaling the output.
        lora_dropout (float): LoRA's dropout rate.
        start_block (int): first block to start replacing the attention layers.
    """

    for block in transformer.resblocks[start_block:]:
        attn = block.attn
        embed_dim = attn.embed_dim
        num_heads = attn.num_heads
        dropout = attn.dropout
        lora_attn = LoraMultiheadAttention(
            embed_dim=embed_dim,
            torch_tgt_module=attn,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            num_heads=num_heads,
            dropout=dropout,
        )
        block.attn = lora_attn

    return transformer




class LoRALinear(nn.Module):
    def __init__(self, torch_tgt_module, lora_r: int,lora_alpha: int,lora_dropout: float,in_features: int,out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = lora.Linear(
            in_features=in_features,
            out_features=out_features,
            r=lora_r,
            lora_alpha=lora_alpha,
        )
        self.set_parameters(torch_tgt_module)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
        

    def set_parameters(self, torch_tgt_module):
        assert isinstance(torch_tgt_module,nn.Linear)
        assert self.in_features == torch_tgt_module.in_features
        assert self.out_features == torch_tgt_module.out_features
        self.linear.weight.data = torch_tgt_module.weight.data
        self.linear.bias.data = torch_tgt_module.bias.data



def lora_replace_linear_layers(
    transformer: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    start_block: int = 0,
):
    """
    Utility function to replace linear layers of a CLIP transformer model
    with LoRAlinear. It expects a pre-defined structure (following OpenAI's CLIP).

    Args:
        transformer (nn.Module): transformer to replace linear layers.
        lora_r (int): LoRA's rank.
        lora_alpha (int): LoRA's alpha for scaling the output.
        lora_dropout (float): LoRA's dropout rate.
        start_block (int): first block to start replacing the linear layers.
    """

    for block in transformer.resblocks[start_block:]:
        c_fc = block.mlp.c_fc
        c_proj = block.mlp.c_proj
        in_features1 = c_fc.in_features
        out_features1 = c_fc.out_features
        in_features2 = c_proj.in_features
        out_features2 = c_proj.out_features
        lora_c_fc = LoRALinear(
            torch_tgt_module=c_fc,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            in_features=in_features1,
            out_features=out_features1,
        )
        lora_c_proj = LoRALinear(
            torch_tgt_module=c_proj,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            in_features=in_features2,
            out_features=out_features2,
        )
        block.mlp.c_fc = lora_c_fc
        block.mlp.c_proj = lora_c_proj

    return transformer

