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


from timm.models.vision_transformer import Attention, Mlp, Block
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List


softmax = torch.nn.Softmax(dim=0)


def select_op(op_params, num_ops):
    prob = softmax(op_params)
    op_ids = torch.multinomial(prob, num_ops, replacement=True).tolist()
    return op_ids


def trace_prob(op_params, op_ids):
    probs = softmax(op_params)
    tp = 1
    for idx in op_ids:
        tp = tp * probs[idx]
    return tp


class RWAug_Search:
    def __init__(self, n, idxs):
        self.n = n
        # idxs is the operation id
        self.idxs = idxs
        self.op_list = op_list

    def __call__(self, img):
        assert len(self.idxs) == self.n
        # print(self.idxs)
        for idx in self.idxs:
            img = op_list[idx](img)
        return img


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


op_list = [
    Wrapped_Attention_forward_REST_Attack,
    Wrapped_Attention_forward_Sparse_Attack,
    Wrapped_Attention_forward_Shuffle_Attack,
    Wrapper_FFN_forward_MoE_Attack,
]

# op_list = [Wrapped_Attention_forward_Sparse_Attack,Wrapped_Attention_forward_Sparse_Attack,Wrapped_Attention_forward_Sparse_Attack]


class L2S(Attack):
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
        epoch=20,
        decay=1.0,
        num_scale=3,
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
        self.num_scale = num_scale

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

    def wrap_attention(self, model, selected_op):
        # transform the attention module in the model to the wrapped attention module
        for name, module in model.named_modules():
            # if isinstance(module, Attention):
            #     module.forward = selected_op.__get__(module)
            # if isinstance(module, Mlp):
            #     module.forward = selected_op.__get__(module)
            # if selected_op is Wrapper_FFN_forward_MoE_Attack, then modify the Mlp module
            if selected_op == Wrapper_FFN_forward_MoE_Attack:
                if isinstance(module, Mlp):
                    module.forward = selected_op.__get__(module)
            elif selected_op in [
                Wrapped_Attention_forward_REST_Attack,
                Wrapped_Attention_forward_Sparse_Attack,
                Wrapped_Attention_forward_Shuffle_Attack,
            ]:
                if isinstance(module, Attention):
                    module.named_id = 0
                    module.forward = selected_op.__get__(module)
            else:
                raise ValueError(f"Unsupported selected_op: {selected_op}")

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
            torch.zeros(aug_length, requires_grad=True), requires_grad=True
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

        for e in range(self.epoch):
            # transform data
            aug_probs = []
            losses = []

            for i in range(self.num_scale):
                rw_search = RWAug_Search(ops_num, [0, 0])

                augtype = (ops_num, select_op(aug_param, ops_num))
                aug_prob = trace_prob(aug_param, augtype[1])
                rw_search.n = augtype[0]
                rw_search.idxs = augtype[1]

                print(rw_search.idxs)

                aug_probs.append(aug_prob)

                # logits_merge = 0.
                for seletec_op in rw_search.idxs:
                    self.wrap_attention(self.model, op_list[seletec_op])
                    logits = self.get_logits(self.transform(data + delta))
                    # logits_merge += logits

                    # logits = logits_merge / len(rw_search.idxs)
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

        # print(softmax(aug_param))
        # print(aug_param)
        return delta.detach()
