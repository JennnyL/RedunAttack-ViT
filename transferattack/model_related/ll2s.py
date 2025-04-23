from typing import Any
import math
import torch
import random
from PIL import ImageOps
from ..utils import *
from ..attack import Attack
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import Dropout
import copy
import pdb


from timm.models.vision_transformer import (
    Attention,
    Mlp,
    Block,
    VisionTransformer,
)

from timm.models import (
    PoolingVisionTransformer,
)


from timm.models.swin_transformer import WindowAttention, SwinTransformer
from timm.models.swin_transformer import Mlp as SwinFFN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List


softmax = torch.nn.Softmax(dim=-1)


def select_op(op_params, num_ops):
    prob = softmax(op_params)
    op_ids = torch.multinomial(prob, num_ops, replacement=True).tolist()
    return op_ids


def trace_prob(op_params, op_ids):
    probs = softmax(op_params)  # shape: (n_layers, n_ops)
    layer_indices = torch.arange(len(op_ids))[:, None]  # shape: (n_layers, 1)
    selected_probs = probs[layer_indices, op_ids]  # shape: (n_layers, n_sampled_ops)
    tp = torch.prod(selected_probs)
    return tp


class RWAug_Search:
    def __init__(self, n, idxs):
        self.n = n
        # idxs is the operation id
        self.idxs = idxs
        self.op_list = op_list


q_rest = {}
k_rest = {}
v_rest = {}
rest_p = 0.3


def Wrapped_Attention_forward_REST_Attack(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, self.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    named_id = self.named_id
    global q_rest, k_rest, v_rest
    num_tokens = q.shape[2]
    filling = False
    if named_id in q_rest:
        # concatenate the q, k, v
        filling = True
        q = torch.cat([q, q_rest[named_id]], dim=2)
        k = torch.cat([k, k_rest[named_id]], dim=2)
        v = torch.cat([v, v_rest[named_id]], dim=2)
    else:
        global rest_p
        sample_num_tokens = int(rest_p * num_tokens)
        num_heads = q.shape[1]
        selected_token_ids = [
            torch.from_numpy(
                np.random.choice(
                    torch.arange(1, num_tokens), sample_num_tokens, replace=False
                )
            )
            for _ in range(num_heads)
        ]
        selected_token_ids = (
            torch.stack(selected_token_ids, dim=0).unsqueeze(0).expand(B, -1, -1)
        )
        batch_indices = (
            torch.arange(B).view(B, 1, 1).expand(-1, num_heads, sample_num_tokens)
        )
        head_indices = (
            torch.arange(num_heads)
            .view(1, num_heads, 1)
            .expand(B, -1, sample_num_tokens)
        )

        q_rest[named_id] = q[batch_indices, head_indices, selected_token_ids]
        k_rest[named_id] = k[batch_indices, head_indices, selected_token_ids]
        v_rest[named_id] = v[batch_indices, head_indices, selected_token_ids]
    q, k = self.q_norm(q), self.k_norm(k)
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = attn @ v
    if filling:
        x = x[:, :, :num_tokens]
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def Wrapped_WindowAttention_forward_REST_Attack(
    self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        x: input features with shape of (num_windows*B, N, C)
        mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
    """
    B_, N, C = x.shape
    B = B_
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    named_id = self.named_id
    global q_rest, k_rest, v_rest
    num_tokens = q.shape[2]
    filling = False
    if named_id in q_rest:
        # concatenate the q, k, v
        filling = True
        q = torch.cat([q, q_rest[named_id]], dim=2)
        k = torch.cat([k, k_rest[named_id]], dim=2)
        v = torch.cat([v, v_rest[named_id]], dim=2)
    else:
        global rest_p
        sample_num_tokens = int(rest_p * num_tokens)
        num_heads = q.shape[1]
        selected_token_ids = [
            torch.from_numpy(
                np.random.choice(
                    torch.arange(1, num_tokens), sample_num_tokens, replace=False
                )
            )
            for _ in range(num_heads)
        ]
        selected_token_ids = (
            torch.stack(selected_token_ids, dim=0).unsqueeze(0).expand(B, -1, -1)
        )
        batch_indices = (
            torch.arange(B).view(B, 1, 1).expand(-1, num_heads, sample_num_tokens)
        )
        head_indices = (
            torch.arange(num_heads)
            .view(1, num_heads, 1)
            .expand(B, -1, sample_num_tokens)
        )

        q_rest[named_id] = q[batch_indices, head_indices, selected_token_ids]
        k_rest[named_id] = k[batch_indices, head_indices, selected_token_ids]
        v_rest[named_id] = v[batch_indices, head_indices, selected_token_ids]

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn + self._get_rel_pos_bias()
    if mask is not None:
        num_win = mask.shape[0]
        attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(
            1
        ).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = self.softmax(attn)
    attn = self.attn_drop(attn)
    x = attn @ v

    if filling:
        x = x[:, :, :num_tokens]

    x = x.transpose(1, 2).reshape(B_, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


sparse_p = 0.4


def Wrapped_Attention_forward_Sparse_Attack(self, x: torch.Tensor) -> torch.Tensor:
    global sparse_p
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
    attn = attn * (torch.rand_like(attn) > sparse_p).float()
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def Wrapped_WindowAttention_forward_Sparse_Attack(
    self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    global sparse_p
    """
    Args:
        x: input features with shape of (num_windows*B, N, C)
        mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
    """
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn + self._get_rel_pos_bias()
    if mask is not None:
        num_win = mask.shape[0]
        attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(
            1
        ).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = self.softmax(attn)
    attn = self.attn_drop(attn)

    attn = attn * (torch.rand_like(attn) > sparse_p).float()

    x = attn @ v
    x = x.transpose(1, 2).reshape(B_, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


shuffle_head_prob = 0.5
shuffle_head_ratio = 0.45


def Wrapped_Attention_forward_Shuffle_Attack(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, self.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    if torch.rand(1) < shuffle_head_prob:
        # random shuffle the attention weights of different heads, along the second dimension
        num_heads = attn.shape[1]
        shuffled_num_heads = int(num_heads * shuffle_head_ratio)
        head_indices = torch.randperm(num_heads)[:shuffled_num_heads]
        ordered_head_indices = torch.sort(head_indices)[0]
        copy_attn = attn.clone()
        copy_attn[:, head_indices, :] = attn[:, ordered_head_indices, :]
        attn = copy_attn.clone()

    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def Wrapped_WindowAttention_forward_Shuffle_Attack(
    self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        x: input features with shape of (num_windows*B, N, C)
        mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
    """
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn + self._get_rel_pos_bias()
    if mask is not None:
        num_win = mask.shape[0]
        attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(
            1
        ).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = self.softmax(attn)
    attn = self.attn_drop(attn)

    if torch.rand(1) < shuffle_head_prob:
        # random shuffle the attention weights of different heads, along the second dimension
        num_heads = attn.shape[1]
        shuffled_num_heads = int(num_heads * shuffle_head_ratio)
        head_indices = torch.randperm(num_heads)[:shuffled_num_heads]
        ordered_head_indices = torch.sort(head_indices)[0]
        copy_attn = attn.clone()
        copy_attn[:, head_indices, :] = attn[:, ordered_head_indices, :]
        attn = copy_attn.clone()

    x = attn @ v
    x = x.transpose(1, 2).reshape(B_, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


moe_N = 5
moe_prob = 0.3


def Wrapper_FFN_forward_MoE_Attack(self, input):
    output = 0.0
    global moe_N
    global moe_prob
    current_N = np.random.randint(2, moe_N + 1)
    for n in range(current_N):
        x = self.fc1(input)
        x = self.act(x)
        x = x * (torch.rand_like(x) > moe_prob).float()
        # x = self.drop1(x)
        x = self.fc2(x)
        # x = self.drop2(x)
        output += x
    output = output / current_N
    return output


def Wrapper_SwinFFN_forward_MoE_Attack(self, input):
    output = 0.0
    global moe_N
    global moe_prob
    current_N = np.random.randint(2, moe_N + 1)
    for n in range(current_N):
        x = self.fc1(input)
        x = self.act(x)
        x = self.norm(x)
        x = x * (torch.rand_like(x) > moe_prob).float()
        # x = self.drop1(x)
        x = self.fc2(x)
        # x = self.drop2(x)
        output += x
    output = output / current_N
    return output


opt_tokens = None


def forward_features(self, x):
    x = self.patch_embed(x)
    x = self._pos_embed(x)

    if opt_tokens is not None:
        x = torch.cat([x, opt_tokens], dim=1)
    else:
        print("no robust tokens")
        x = x

    x = self.norm_pre(x)
    x = self.blocks(x)
    x = self.norm(x)
    # print("-"*50)
    # print(x.shape,opt_tokens.shape)

    # print(x.shape)
    # print("-"*50)
    return x


def forward_Swin_features(self, x):
    x = self.patch_embed(x)
    if opt_tokens is not None:
        x = torch.cat([x, opt_tokens], dim=1)
        # import ipdb

        # ipdb.set_trace()
        # x = torch.cat([x, torch.randn((len(x), 20, 56, 96))], dim=1)
        # x = torch.cat([x, torch.randn((len(x), 76, 20, 96))], dim=2)
        # ipdb.set_trace()
        print("catting swin tokens")
    else:
        x = x
    x = self.layers(x)
    x = self.norm(x)
    return x


def forward_PiT_features(self, x):
    x = self.patch_embed(x)
    x = self.pos_drop(x + self.pos_embed)

    # import ipdb

    # ipdb.set_trace()

    cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    x, cls_tokens = self.transformers((x, cls_tokens))
    cls_tokens = self.norm(cls_tokens)

    # if opt_tokens is not None:
    #     import ipdb

    #     ipdb.set_trace()
    #     x = torch.cat([x, opt_tokens], dim=1)
    #     print("catting pit tokens")
    # else:
    #     x = x
    return cls_tokens


op_list = [
    Wrapped_Attention_forward_REST_Attack,
    Wrapped_Attention_forward_Sparse_Attack,
    Wrapped_Attention_forward_Shuffle_Attack,
    Wrapper_FFN_forward_MoE_Attack,
]


swin_list = [
    Wrapped_WindowAttention_forward_REST_Attack,
    Wrapped_WindowAttention_forward_Sparse_Attack,
    Wrapped_WindowAttention_forward_Shuffle_Attack,
    Wrapper_SwinFFN_forward_MoE_Attack,
]

# op_list = [Wrapped_Attention_forward_Sparse_Attack,Wrapped_Attention_forward_Sparse_Attack,Wrapped_Attention_forward_Sparse_Attack]


class LL2S(Attack):
    """
    L2T Attack
    'Learning to Transform Dynamically for Better Adversarial Transferability'(https://arxiv.org/abs/2405.14077)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of scales for input transformation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1, num_scale=3

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/l2t/resnet18 --attack l2t --model=resnet18 --batchsize 2
        python main.py --input_dir ./path/to/data --output_dir adv_data/l2t/resnet18 --eval
    """

    def __init__(
        self,
        model_name,
        epsilon=16 / 255,
        alpha=1.6 / 255,
        epoch=10,
        decay=1.0,
        num_scale=10,
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=None,
        attack="L2T",
        **kwargs,
    ):
        super().__init__(
            attack, model_name, epsilon, targeted, random_start, norm, loss, device
        )
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_scale = 10
        self.model_name = model_name

        if "swin" in model_name:
            attention_modules, ffn_modules = self.enumerate_swin_module(self.model)
        else:
            attention_modules, ffn_modules = self.enumerate_module(self.model)
        self.attention_modules = attention_modules
        self.ffn_modules = ffn_modules
        self.num_attention = len(attention_modules)
        self.num_ffn = len(ffn_modules)
        assert (
            self.num_attention == self.num_ffn
        ), "The number of attention modules and ffn modules should be the same"
        self.num_layers = self.num_attention

        if "swin" in model_name:
            self.model = self.wrap_forward_swin_features(self.model)
            self.token_dim = 96
        elif "pit" in model_name:
            self.model = self.wrap_forward_PiT_features(self.model)
            self.token_dim = 768
        else:
            self.model = self.wrap_forward_features(self.model)
            self.token_dim = 768

        self._model_name_ = model_name
        assert os.environ.get("NUM_ROBUST_TOKENS", None) is not None
        self.num_tokens = int(os.environ.get("NUM_ROBUST_TOKENS", None))
        assert os.environ.get("ROBUST_TOKENS_TYPE", None) is not None
        self.robust_tokens_type = os.environ.get("ROBUST_TOKENS_TYPE", None)
        assert self.robust_tokens_type in ["dynamic", "global", "none"]
        self.prompt_learning_alpha = (
            1e-2  # learning rate for updating dynamic robust tokens
        )
        self.dynamic_robust_epoch = 1

    def init_robust_delta(self, N):
        # delta = torch.rand(self.num_tokens, self.num_patches).to(self.device)
        # delta = torch.zeros(N, self.num_tokens, self.token_dim).to(self.device)
        if "swin" in self._model_name_:
            s = int(np.sqrt(self.num_tokens))
            delta = torch.randn((N, s, s, self.token_dim)).to(self.device) * 10
        elif "pit" in self._model_name_:
            delta = (
                torch.randn((N, self.num_tokens, self.token_dim)).to(self.device) * 10
            )
        else:
            delta = (
                torch.randn((N, self.num_tokens, self.token_dim)).to(self.device) * 10
            )
        delta.requires_grad = True
        return delta

    def update_robust_delta(self, delta, grad, **kwargs):
        # grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
        # scaled_grad = grad # / (grad_norm + 1e-20)
        delta = delta - grad.sign() * self.prompt_learning_alpha
        # delta = delta - grad * self.prompt_learning_alpha
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

    def wrap_forward_swin_features(self, model):

        # assert the class of  model is VisionTransformer
        # import pdb;pdb.set_trace()
        # assert isinstance(model[1], VisionTransformer)
        #
        # model.forward_features = forward_features.__get__(model)
        # return model

        for name, module in model.named_modules():
            if isinstance(module, SwinTransformer):
                # import pdb;pdb.set_trace()
                module.forward_features = forward_Swin_features.__get__(module)
                return model
        # import pdb;pdb.set_trace()
        raise Exception("The model does not contain SwinVisionTransformer module")

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

    def get_loss(self, logits, label, num_copy):
        """
        The loss calculation, which should be overrideen when the attack change the loss calculation (e.g., ATA, etc.)
        """
        # Calculate the loss
        return (
            -self.loss(logits, label.repeat(num_copy))
            if self.targeted
            else self.loss(logits, label.repeat(num_copy))
        )

    def get_grad(self, loss, delta, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[
            0
        ]

    def enumerate_module(self, model):
        ffn_modules = []
        attention_modules = []
        # import pdb;pdb.set_trace()
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                module.original_forward = module.forward
                module.named_id = 0
                attention_modules.append((name, module))
            elif isinstance(module, Mlp):
                module.original_forward = module.forward
                module.named_id = 0
                ffn_modules.append((name, module))
        return attention_modules, ffn_modules

    def enumerate_swin_module(self, model):
        ffn_modules = []
        attention_modules = []
        # import pdb;pdb.set_trace()
        for name, module in model.named_modules():
            if isinstance(module, WindowAttention):
                module.original_forward = module.forward
                module.named_id = 0
                attention_modules.append((name, module))
            elif isinstance(module, SwinFFN):
                module.original_forward = module.forward
                module.named_id = 0
                ffn_modules.append((name, module))
        return attention_modules, ffn_modules

    def wrap_attention(self, model, selected_op_idx_list):
        for layer_idx in range(self.num_layers):
            selected_op = op_list[selected_op_idx_list[layer_idx]]
            if selected_op in [Wrapper_FFN_forward_MoE_Attack]:
                self.ffn_modules[layer_idx][1].forward = selected_op.__get__(
                    self.ffn_modules[layer_idx][1]
                )
            elif selected_op in [
                Wrapped_Attention_forward_REST_Attack,
                Wrapped_Attention_forward_Sparse_Attack,
                Wrapped_Attention_forward_Shuffle_Attack,
            ]:
                self.attention_modules[layer_idx][1].forward = selected_op.__get__(
                    self.attention_modules[layer_idx][1]
                )
            else:
                raise ValueError(f"Unsupported operation: {selected_op}")

    def wrap_swin_attention(self, model, selected_op_idx_list):
        for layer_idx in range(self.num_layers):
            selected_op = swin_list[selected_op_idx_list[layer_idx]]
            if selected_op in [Wrapper_SwinFFN_forward_MoE_Attack]:
                self.ffn_modules[layer_idx][1].forward = selected_op.__get__(
                    self.ffn_modules[layer_idx][1]
                )
            elif selected_op in [
                Wrapped_WindowAttention_forward_REST_Attack,
                Wrapped_WindowAttention_forward_Sparse_Attack,
                Wrapped_WindowAttention_forward_Shuffle_Attack,
            ]:
                self.attention_modules[layer_idx][1].forward = selected_op.__get__(
                    self.attention_modules[layer_idx][1]
                )
            else:
                raise ValueError(f"Unsupported operation: {selected_op}")

    def cleanup(self):
        """
        Clean up the model after the attack
        """
        for layer_idx in range(self.num_layers):
            self.attention_modules[layer_idx][1].forward = self.attention_modules[
                layer_idx
            ][1].original_forward
            self.ffn_modules[layer_idx][1].forward = self.ffn_modules[layer_idx][
                1
            ].original_forward

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure
        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        aug_length = len(op_list)
        ops_num = 2
        learning_rate = 0.01
        # self.num_scale = 10
        aug_param = torch.nn.Parameter(
            torch.zeros(self.num_layers, aug_length, requires_grad=True),
            requires_grad=True,
        )
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        momentum = 0

        global q_rest, k_rest, v_rest
        q_rest = {}
        k_rest = {}
        v_rest = {}

        if self.robust_tokens_type == "global":
            tensor_filepath = "/home/cxu-serve/p62/zwang236/ViT_Robustness/data/attack_image_robust_riter1_img50000_ensemble_400_tokens.pt"
            robust_tokens = (
                torch.load(tensor_filepath)
                .to(self.device)
                .unsqueeze(0)
                .repeat([data.shape[0], 1, 1])
                .clone()
            )
        elif self.robust_tokens_type == "dynamic":
            momentum_robust = 0.0
            robust_tokens = self.init_robust_delta(len(data)).to(self.device)
        else:
            assert self.robust_tokens_type == "none"
            robust_tokens = None
        global opt_tokens

        for e in range(self.epoch):
            opt_tokens = (
                robust_tokens.clone().detach() if robust_tokens is not None else None
            )
            # transform data
            aug_probs = []
            losses = []

            for i in range(self.num_scale):
                rw_search = RWAug_Search(ops_num, [0, 0])

                augtype = (ops_num, np.array(select_op(aug_param, ops_num)))
                prob = 1.0
                for ops_index in range(ops_num):
                    # import pdb;pdb.set_trace()
                    aug_prob = trace_prob(aug_param, augtype[1][:, ops_index])
                    # prob *= aug_prob
                    aug_probs.append(aug_prob)
                # aug_prob = prob
                rw_search.n = augtype[0]
                rw_search.idxs = augtype[1]

                # print(rw_search.idxs)

                # aug_probs.append(aug_prob)
                # mean_logits = 0.
                for ops_index in range(ops_num):
                    selected_ops = rw_search.idxs[:, ops_index]
                    self.cleanup()
                    if "swin" in self.model_name:
                        self.wrap_swin_attention(self.model, selected_ops)
                    else:
                        self.wrap_attention(self.model, selected_ops)
                    logits = self.get_logits(self.transform(data + delta))
                    # mean_logits += logits
                    # mean_logits = mean_logits/ops_num
                    # logits = mean_logits
                    losses.append(
                        self.get_loss(
                            logits, label, math.floor((len(logits) + 0.01) / len(label))
                        ).reshape(1)
                    )

            # Calculate the loss
            loss = torch.sum(torch.cat(losses)) / self.num_scale

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            aug_losses = torch.cat(
                [aug_probs[i] * losses[i].reshape(1) for i in range(self.num_scale)]
            )
            aug_loss = torch.sum(aug_losses) / self.num_scale

            aug_grad = torch.autograd.grad(
                aug_loss, aug_param, retain_graph=False, create_graph=False
            )[0]
            aug_param = aug_param + learning_rate * aug_grad
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

            if self.robust_tokens_type == "dynamic":
                for _ in range(self.dynamic_robust_epoch):
                    opt_tokens = robust_tokens
                    robust_logits = self.get_logits(self.transform(data + delta))
                    robust_loss = self.get_loss(
                        logits=robust_logits, label=label, num_copy=1
                    )
                    robust_grad = self.get_grad(robust_loss, robust_tokens)
                    momentum_robust = self.get_robust_momentum(
                        robust_grad, momentum=momentum_robust
                    )
                    robust_tokens = self.update_robust_delta(
                        robust_tokens, momentum_robust
                    )

        # print(softmax(aug_param))
        # print(aug_param)
        return delta.detach()
