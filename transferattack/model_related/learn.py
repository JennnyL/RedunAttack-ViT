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
    VisionTransformer,
)

from timm.models import (
    PoolingVisionTransformer,
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

    if opt_tokens is not None:
        x = torch.cat([x, opt_tokens], dim=1)
    else:
        x = x

    x = self.norm_pre(x)
    x = self.blocks(x)
    x = self.norm(x)
    return x


def forward_PiT_features(self, x):
    x = self.patch_embed(x)
    x = self.pos_drop(x + self.pos_embed)

    cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    if opt_tokens is not None:
        mat_size = x.shape[2]
        x = torch.cat(
            [
                x,
                opt_tokens[:, :, :mat_size, :],
            ],
            dim=-1,
        )
        x = torch.cat(
            [x, opt_tokens[:, :, mat_size:, :].permute(0, 1, 3, 2)],
            dim=-2,
        )
    else:
        x = x
    x, cls_tokens = self.transformers((x, cls_tokens))
    cls_tokens = self.norm(cls_tokens)

    return cls_tokens


class LearnAttack(Attack):
    def __init__(
        self,
        model_name,
        epsilon=16 / 255,
        alpha=1.6 / 255,
        epoch=10,
        decay=1.0,
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=None,
        attack="GI-FGSM",
        **kwargs,
    ):
        assert kwargs is not None

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
        if "swin" in model_name:
            self.model = self.wrap_forward_swin_features(self.model)
            self.token_dim = 96
        elif "pit" in model_name:
            self.model = self.wrap_forward_PiT_features(self.model)
            self.token_dim = 256
        else:
            self.model = self.wrap_forward_features(self.model)
            self.token_dim = 768

        self._model_name_ = model_name
        assert os.environ.get("NUM_ROBUST_TOKENS", None) is not None
        self.num_tokens = int(os.environ.get("NUM_ROBUST_TOKENS", None))
        assert os.environ.get("ROBUST_TOKENS_TYPE", None) is not None
        self.robust_tokens_type = os.environ.get("ROBUST_TOKENS_TYPE", None)
        assert self.robust_tokens_type in ["dynamic", "dynamic_iter", "global", "none"]
        self.num_tokens_use_ratio = 1
        self.prompt_learning_alpha = (
            1e-2  # learning rate for updating dynamic robust tokens
        )
        self.dynamic_robust_epoch = 1

    def init_robust_delta(self, N):
        if "swin" in self._model_name_:
            s = int(np.sqrt(self.num_tokens))
            delta = torch.randn((N, s, s, self.token_dim)).to(self.device) * 10
            delta.requires_grad = True
        elif "pit" in self._model_name_:
            m_size = 31
            margin_size = int(
                (-2 * m_size + np.sqrt(2 * m_size * 2 * m_size + 4 * self.num_tokens))
                / 2
            )
            if True:
                delta = (
                    torch.randn(
                        (N, self.token_dim, m_size + m_size + margin_size, margin_size)
                    ).to(self.device)
                    * 10
                )
            else:
                delta = torch.zeros(
                    (N, self.token_dim, m_size + m_size + margin_size, margin_size)
                ).to(self.device)
            delta.requires_grad = True
        else:
            delta = (
                torch.randn((N, self.num_tokens, self.token_dim)).to(self.device) * 10
            )
            delta.requires_grad = True
        return delta

    def update_robust_delta(self, delta, grad, **kwargs):
        delta = delta - grad.sign() * self.prompt_learning_alpha
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

    def wrap_forward_PiT_features(self, model):
        # assert the class of  model is VisionTransformer
        # import pdb;pdb.set_trace()
        # assert isinstance(model[1], VisionTransformer)
        #
        # model.forward_features = forward_features.__get__(model)
        # return model
        for name, module in model.named_modules():
            if isinstance(module, PoolingVisionTransformer):
                # import pdb;pdb.set_trace()
                module.forward_features = forward_PiT_features.__get__(module)
                return model
        # import pdb;pdb.set_trace()
        raise Exception("The model does not contain PoolingVisionTransformer module")

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
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)  # shape: (batch_size, c, h, w)
        label = label.clone().detach().to(self.device)  # shape: (batch_size, )

        global opt_tokens
        if self.robust_tokens_type == "global":
            # tensor_filepath = f"./data/attack_image_robust_riter1_img50000_ensemble_{self.num_tokens}_tokens_randn_init_all.pt"
            # tensor_filepath = f"./data/attack_image_robust_riter1_img50000_ensemble_{self.num_tokens}_tokens_all.pt"
            # tensor_filepath = "./data/new_dymaic_attack_image_robust_riter1_img1000_ensemble_400/aa_merged.pt"
            # tensor_filepath = f"./data/zero_momentum_riter{1}_img{50000}_ensemble_{self.num_tokens}_new.pt"
            tensor_filepath = (
                "./data/pit_b_224_dynamic_zeros_init_1_img1000_ensemble_400_new.pt"
            )
            print(f"successfully loaded {tensor_filepath}")
            if "vit" in self._model_name_:
                robust_tokens = (
                    torch.load(tensor_filepath)
                    .to(self.device)
                    .unsqueeze(0)
                    .repeat([data.shape[0], 1, 1])
                    .clone()
                )
            else:
                robust_tokens = (
                    torch.load(tensor_filepath)
                    .to(self.device)
                    .unsqueeze(0)
                    .repeat([data.shape[0], 1, 1, 1])
                    .clone()
                )
        elif self.robust_tokens_type == "dynamic":
            momentum_robust = 0.0
            robust_tokens = self.init_robust_delta(len(data)).to(self.device)
        elif self.robust_tokens_type == "dynamic_iter":
            assert (
                len(data) == 1
            ), "batch_size should be 1 for dynamic iterative robust tokens"
            momentum_robust = 0.0
            if opt_tokens is None:
                robust_tokens = self.init_robust_delta(len(data)).to(self.device)
            else:
                robust_tokens = opt_tokens.clone().detach()
                robust_tokens.requires_grad = True
                print("reuse the robust tokens of the first batch")
        else:
            assert self.robust_tokens_type == "none"
            robust_tokens = None

        momentum = 0.0

        attack_delta = self.init_delta(data).to(self.device)

        robust_max_iter = 1

        for _ in range(self.epoch):  # attack iteration
            # 1. attack part

            # robust_tokens = self.init_robust_delta(len(data)).to(self.device)
            # opt_tokens = robust_tokens.clone().detach()
            if self.robust_tokens_type == "none":
                opt_tokens = None
            else:
                opt_tokens = get_opt_tokens(
                    robust_tokens.clone().detach(), self.num_tokens_use_ratio
                )

            logits = self.get_logits(self.transform(data + attack_delta))
            loss = self.get_loss(logits=logits, label=label)
            attack_grad = self.get_grad(loss, attack_delta)
            momentum = self.get_momentum(attack_grad, momentum=momentum)
            attack_delta = self.update_delta(attack_delta, data, momentum, self.alpha)

            # 2. robustify part, only for dynamic tokens
            if self.robust_tokens_type in ["dynamic", "dynamic_iter"]:
                for _ in range(robust_max_iter):
                    opt_tokens = robust_tokens

                    drop_prob = kwargs["dropout_prob"]
                    mask = torch.rand_like(opt_tokens) >= drop_prob
                    opt_tokens = opt_tokens * mask

                    opt_tokens = opt_tokens / (1 - drop_prob)

                    robust_logits = self.get_logits(self.transform(data + attack_delta))
                    robust_loss = self.get_loss(logits=robust_logits, label=label)
                    robust_grad = self.get_grad(robust_loss, robust_tokens)
                    if self.robust_tokens_type == "dynamic":
                        momentum_robust = self.get_robust_momentum(
                            robust_grad, momentum=momentum_robust
                        )
                    elif self.robust_tokens_type == "dynamic_iter":
                        momentum_robust = self.get_robust_momentum(
                            robust_grad, momentum=0
                        )

                    robust_tokens = self.update_robust_delta(
                        robust_tokens, momentum_robust
                    )

        # only save tokens for dynamic tokens
        if self.robust_tokens_type in ["dynamic", "dynamic_iter"]:
            tensor_filepath = os.path.join(
                "./data/",
                f"{self._model_name_}_{self.robust_tokens_type}_rand_init_{robust_max_iter}_img{kwargs['total_images_num']}_ensemble_{self.num_tokens}_new.pt",
            )
            r_tokens = robust_tokens.clone().detach().cpu()

            if self.robust_tokens_type == "dynamic":
                r_tokens = r_tokens.sum(dim=0) / kwargs["total_images_num"]
                if os.path.exists(tensor_filepath):
                    tokens = torch.load(tensor_filepath)
                    r_tokens += tokens

            torch.save(r_tokens, tensor_filepath)

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


def get_opt_tokens(robust_tokens, num_tokens_use_ratio, keep_order=True):
    if num_tokens_use_ratio >= 1:
        return robust_tokens

    sampled_robust_tokens = []
    for ii in range(len(robust_tokens)):
        samples = random.sample(
            range(0, robust_tokens.shape[1]),
            int(num_tokens_use_ratio * robust_tokens.shape[1]),
        )
        if keep_order:
            samples = sorted(samples)
        sampled_robust_tokens.append(robust_tokens[ii, samples].unsqueeze(0))
    return torch.cat(sampled_robust_tokens, dim=0).clone().detach()
