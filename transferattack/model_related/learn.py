import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from timm.models.vision_transformer import (
    Attention,
    Mlp,
    Block,
    checkpoint_seq,
    VisionTransformer,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List


import torch

from ..utils import *
from ..attack import Attack


opt_tokens = None


def forward_features(self, x):
    x = self.patch_embed(x)
    x = self._pos_embed(x)

    # opt_token shape: (T, D)
    # weight = F.softmax(opt_tokens, dim=-1)
    # import pdb;pdb.set_trace()
    if opt_tokens is not None:
        # append_token = torch.einsum('tn,bnd->btd', opt_tokens, x[:,1:])
        # x = torch.cat([x, append_token], dim=1)
        x = torch.cat([x, opt_tokens], dim=1)
    else:
        x = x

    x = self.norm_pre(x)
    # if self.grad_checkpointing and not torch.jit.is_scripting():
    #     x = checkpoint_seq(self.blocks, x)
    # else:
    x = self.blocks(x)
    x = self.norm(x)
    # print("-"*50)
    # print(x.shape,opt_tokens.shape)

    # print(x.shape)
    # print("-"*50)
    return x


class LearnAttack(Attack):
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
        self.pre_epoch = 3
        self.num_tokens = 400
        self.token_dim = 768
        self.num_patches = 196
        self.prompt_learning_alpha = 1e-3
        self.model = self.wrap_forward_features(self.model)

    def init_robust_delta(self, N):
        # delta = torch.rand(self.num_tokens, self.num_patches).to(self.device)
        delta = torch.zeros(N, self.num_tokens, self.token_dim).to(self.device)
        delta.requires_grad = True
        return delta

    def update_robust_delta(self, delta, grad, **kwargs):
        # grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
        # scaled_grad = grad # / (grad_norm + 1e-20)
        delta = delta - grad * self.prompt_learning_alpha
        # import pdb;pdb.set_trace()
        return delta.detach().requires_grad_(True)

    def get_robust_momentum(self, grad, momentum, **kwargs):
        """
        The momentum calculation
        """
        return momentum * self.decay + grad

    def wrap_forward_features(self, model):

        # assert the class of  model is VisionTransformer
        # import pdb;pdb.set_trace()
        # assert isinstance(model[1], VisionTransformer)
        #
        # model.forward_features = forward_features.__get__(model)
        # return model

        for name, module in model.named_modules():
            if isinstance(module, VisionTransformer):
                # import pdb;pdb.set_trace()
                module.forward_features = forward_features.__get__(module)
                return model
        # import pdb;pdb.set_trace()
        raise Exception("The model does not contain VisionTransformer module")

    def init_delta(self, data, **kwargs):
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == "linfty":
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0, 1).to(self.device)
                delta *= r / n * self.epsilon
            delta = clamp(delta, img_min - data, img_max - data)
        delta.requires_grad = True
        return delta

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """

        # assert len(data) == 1, "Only support batch_size = 1"

        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)  # shape: (batch_size, c, h, w)
        label = label.clone().detach().to(self.device)  # shape: (batch_size, )

        tensor_filepath = (
            "./data/attack_image_robust_riter1_img50000_ensemble_400_tokens.pt"
        )
        robust_tokens = (
            torch.load(tensor_filepath)
            .to(self.device)
            .unsqueeze(0)
            .repeat([data.shape[0], 1, 1])
        )
        global opt_tokens

        momentum = 0.0

        attack_delta = self.init_delta(data).to(self.device)

        for attack_idx in range(self.epoch):  # attack iteration
            # 1. attack part
            # opt_tokens = None
            opt_tokens = robust_tokens.clone().detach()
            logits = self.get_logits(self.transform(data + attack_delta))
            loss = self.get_loss(logits=logits, label=label)
            attack_grad = self.get_grad(loss, attack_delta)
            momentum = self.get_momentum(attack_grad, momentum=momentum)
            attack_delta = self.update_delta(attack_delta, data, momentum, self.alpha)

        return attack_delta.detach()

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