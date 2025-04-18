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
        assert kwargs is not None
        assert kwargs.get("num_tokens", None) is not None
        num_tokens = kwargs.get("num_tokens", None)
        kwargs.pop("num_tokens")

        assert kwargs.get("num_tokens_use_ratio", None) is not None
        num_tokens_use_ratio = kwargs.get("num_tokens_use_ratio", None)
        kwargs.pop("num_tokens_use_ratio")
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

        self.num_tokens = num_tokens
        self.num_tokens_use_ratio = num_tokens_use_ratio

        self.token_dim = 768
        self.num_patches = 196
        self.prompt_learning_alpha = 1e-2
        self.model = self.wrap_forward_features(self.model)

    def init_robust_delta(self, N):
        delta = torch.randn((N, self.num_tokens, self.token_dim)).to(self.device) * 10
        delta.requires_grad = True
        return delta

    def update_robust_delta(self, delta, grad, **kwargs):
        # grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
        # scaled_grad = grad # / (grad_norm + 1e-20)

        # delta = delta - grad.sign() * self.prompt_learning_alpha
        # import ipdb

        # ipdb.set_trace()
        delta = delta - grad.sign() * self.prompt_learning_alpha

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

        global opt_tokens
        if kwargs["load_tokens"]:
            # tensor_filepath = f"./data/attack_image_robust_riter1_img50000_ensemble_{self.num_tokens}_tokens_randn_init_all.pt"
            # tensor_filepath = f"./data/attack_image_robust_riter1_img50000_ensemble_{self.num_tokens}_tokens_all.pt"
            # tensor_filepath = "./data/new_dymaic_attack_image_robust_riter1_img1000_ensemble_400/aa_merged.pt"
            tensor_filepath = f"./data/zero_momentum_riter{1}_img{50000}_ensemble_{self.num_tokens}_new.pt"
            print(f"successfully loaded {tensor_filepath}")
            global_tokens = (
                torch.load(tensor_filepath)
                .to(self.device)
                .reshape(-1, self.token_dim)
                .repeat([len(data), 1, 1])  # (n, num_tokens, token_dim)
                .clone()
            )
            # print(global_tokens.shape)
            # tokens = []
            # for _ in range(len(data)):
            #     samples = random.sample(
            #         range(0, len(global_tokens)),
            #         self.num_tokens,
            #     )
            #     tokens.append(global_tokens[samples].unsqueeze(0).clone())
            # robust_tokens = torch.cat(tokens, dim=0).clone().detach()
            robust_tokens = global_tokens.clone().detach()
            print(robust_tokens.shape)
        else:
            if opt_tokens is None:
                robust_tokens = self.init_robust_delta(len(data)).to(self.device)
            else:
                robust_tokens = opt_tokens.clone().detach()
                # robust_tokens = opt_tokens[: data.shape[0]].clone().detach()
                robust_tokens.requires_grad = True
                print("reuse the robust tokens of the first batch")

        momentum = 0.0
        momentum_robust = 0.0

        attack_delta = self.init_delta(data).to(self.device)

        robust_max_iter = 1

        for _ in range(self.epoch):  # attack iteration
            # 1. attack part

            # robust_tokens = self.init_robust_delta(len(data)).to(self.device)
            # opt_tokens = robust_tokens.clone().detach()
            opt_tokens = get_opt_tokens(
                robust_tokens.clone().detach(), self.num_tokens_use_ratio
            )

            logits = self.get_logits(self.transform(data + attack_delta))
            loss = self.get_loss(logits=logits, label=label)
            attack_grad = self.get_grad(loss, attack_delta)
            momentum = self.get_momentum(attack_grad, momentum=momentum)
            attack_delta = self.update_delta(attack_delta, data, momentum, self.alpha)

            # 2. robustify part, only for dynamic tokens
            if not kwargs["load_tokens"]:
                pred_correct_num = []
                for _ in range(robust_max_iter):
                    opt_tokens = robust_tokens

                    drop_prob = kwargs["dropout_prob"]
                    mask = (
                        torch.rand_like(opt_tokens) >= drop_prob
                    )  # 每个元素以 (1 - p) 的概率保留
                    opt_tokens = opt_tokens * mask  # 掩码应用

                    # 可选：为了保持期望值不变，缩放输出
                    opt_tokens = opt_tokens / (1 - drop_prob)

                    robust_logits = self.get_logits(self.transform(data + attack_delta))
                    pred_correct_num.append(
                        (robust_logits.argmax(dim=1) == label).sum().cpu().item()
                    )
                    robust_loss = self.get_loss(logits=robust_logits, label=label)
                    robust_grad = self.get_grad(robust_loss, robust_tokens)
                    momentum_robust = self.get_robust_momentum(robust_grad, momentum=0)
                    robust_tokens = self.update_robust_delta(
                        robust_tokens, momentum_robust
                    )

        # only save tokens for dynamic tokens
        if not kwargs["load_tokens"]:
            # tensor_folder = f"./data/zero_momentum_attack_image_robust_riter{robust_max_iter}_img{kwargs['total_images_num']}_ensemble_{self.num_tokens}/"
            # os.makedirs(tensor_folder, exist_ok=True)
            # tensor_filepath = os.path.join(
            #     tensor_folder, f"batch_{kwargs['batch_idx']}.pt"
            # )

            tensor_filepath = os.path.join(
                "./data/",
                f"zero_momentum_riter{robust_max_iter}_img{kwargs['total_images_num']}_ensemble_{self.num_tokens}_new.pt",
            )
            # if os.path.exists(tensor_filepath):
            #     tokens = torch.load(tensor_filepath)
            #     robust_tokens = torch.cat([tokens, robust_tokens], dim=0)
            torch.save(robust_tokens, tensor_filepath)
            # print("successfully saved tokens")

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
