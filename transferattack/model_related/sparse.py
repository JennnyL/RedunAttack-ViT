import os
import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import torch

from timm.models.vision_transformer import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List


import torch

from ..utils import *
from ..attack import Attack


attn_weights = []
# skip_sparsity_flag =
sparsity_list = []
sparsity_pack = []
drop_rate_scale = 1


def Wrapped_Attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, self.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    # import pdb;pdb.set_trace()

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    # import pdb;pdb.set_trace()
    # random drop 50% of the attention weights

    assert os.environ.get("ATTN_DROP_RATE", None) is not None
    attn_drop_rate = float(os.environ.get("ATTN_DROP_RATE", None))

    assert os.environ.get("DYNAMIC_SPARSITY_FLAG", None) is not None
    dynamic_sparsity_flag = int(os.environ.get("DYNAMIC_SPARSITY_FLAG", None))
    if dynamic_sparsity_flag:
        assert attn.shape[0] == 1

        pp = 0.01
        # sparsity_threshold = (
        #     (pp * (attn.max(-1)[0]).max(-1)[0]).unsqueeze(-1).unsqueeze(-1)
        # )

        sparsity_threshold = pp * torch.mean(attn, dim=[-1, -2], keepdim=True)

        dense_token_map = (  # (BS, heads, post-vision tokens(including dropped), k-tokens)
            attn >= sparsity_threshold
        )
        dense_token_num = (dense_token_map).sum(dim=(-1, -2))  # (BS, heads)
        sparsity_valid_token_num = attn.shape[-1] * attn.shape[-2]
        sparsity = (
            ((sparsity_valid_token_num - dense_token_num) / sparsity_valid_token_num)
            .mean()
            .cpu()
            .item()
        )
        global sparsity_pack
        sparsity_pack.append(sparsity)

        attn_drop_rate_total = attn_drop_rate * 12  # 12 layers
        if len(sparsity_list) != 0:
            layer_weights = ((torch.tensor(sparsity_list[-1])) * 1).exp()
            layer_weights = layer_weights / layer_weights.sum()
            # layer_weights = torch.clip(layer_weights, 0.01, 1)
            layer_idx = len(sparsity_pack) - 1
            attn_drop_rate = layer_weights[layer_idx] * attn_drop_rate_total
            attn_drop_rate = torch.clip(attn_drop_rate, 0.01, 0.9)

        # if len(sparsity_list) > 0:
        #     print(f"last weights: {sparsity_list[-1]}")
        print(
            f"epoch: {len(sparsity_list)}, layer: {len(sparsity_pack)-1}, drop_ratio: {attn_drop_rate}"
        )

    global drop_rate_scale
    attn_drop_rate *= drop_rate_scale
    # print(f"drop_rate: {attn_drop_rate}")
    attn = attn * (torch.rand_like(attn) > attn_drop_rate).float()
    # print(attn.mean())
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class SparseAttack(Attack):
    def __init__(
        self,
        model_name,
        epsilon=16 / 255,
        alpha=1.6 / 255,
        epoch=10,
        decay=1.0,
        resize_rate=1.1,
        diversity_prob=0.5,
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=None,
        attack="GI-FGSM",
        s=10,
        **kwargs,
    ):
        super().__init__(
            attack,
            model_name,
            epsilon,
            targeted,
            random_start,
            norm,
            loss,
            device,
            **kwargs,
        )
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.s = s
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.wrap_attention()

    def wrap_attention(self):
        # transform the attention module in the model to the wrapped attention module
        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                module.forward = Wrapped_Attention_forward.__get__(module)

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        global sparsity_list
        global sparsity_pack
        global drop_rate_scale
        drop_rate_scale = 1
        sparsity_list = []
        momentum = 0.0
        delta = self.init_delta(data).to(self.device)
        for _ in range(self.epoch):
            sparsity_pack = []
            # Obtain the output
            logits = self.get_logits(self.transform(data + delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

            # sparsity_list.append(sparsity_pack)
            drop_rate_scale *= 1.02

        # with open(f"{attn_drop_rate}_sparsity_list.txt", "a") as fl:
        #     fl.write(f"{sparsity_list}\n")
        sparsity_list = []
        sparsity_pack = []

        return delta.detach()

    def get_loss(self, logits, label, attn_weights_benign=None, attn_weights_adv=None):
        """
        The loss calculation, which should be overrideen when the attack change the loss calculation (e.g., ATA, etc.)
        """
        # Calculate the loss
        # import pdb;pdb.set_trace()

        ori_loss = (
            -self.loss(logits, label) if self.targeted else self.loss(logits, label)
        )

        if attn_weights_benign is None or attn_weights_adv is None:
            return ori_loss
        else:
            # maximize the difference between the benign and adversarial attention weights
            loss = 0
            for i in range(len(attn_weights_benign)):
                loss += torch.cosine_similarity(
                    attn_weights_benign[i].flatten(),
                    attn_weights_adv[i].flatten(),
                    dim=0,
                )
            # import pdb;pdb.set_trace()
            loss = loss / len(attn_weights_benign)
            return ori_loss - loss
