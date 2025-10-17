
import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

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


def Wrapped_Attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    # import pdb;pdb.set_trace()
    
    
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    # import pdb;pdb.set_trace()
    global attn_weights
    attn_weights.append(attn)
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
    


class SparseAttack(Attack):    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., resize_rate=1.1, diversity_prob=0.5, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='GI-FGSM',  s=10, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device, **kwargs)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.s = s
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.wrap_attention()
    
    
    def transform(self, x, **kwargs):
        """
        Random transform the input images
        """
        # do not transform the input image
        if torch.rand(1) > self.diversity_prob:
            return x
        
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        # resize the input image to random size
        rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        # randomly add padding
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        # resize the image back to img_size
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)
    
    
    
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
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        with torch.no_grad():
            logits = self.get_logits(data)
            global attn_weights
            attn_weights_benign = attn_weights
            attn_weights = []
        
        # import pdb;pdb.set_trace()
        
        
        momentum = 0.
        delta = self.init_delta(data).to(self.device)
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            # Calculate the loss
            
            # global attn_weights
            attn_weights_adv = attn_weights
            attn_weights = []
            
            # import pdb;pdb.set_trace()
            loss = self.get_loss(logits, label, attn_weights_benign, attn_weights_adv)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
        
        return delta.detach()
    

    def get_loss(self, logits, label, attn_weights_benign=None, attn_weights_adv=None):
        """
        The loss calculation, which should be overrideen when the attack change the loss calculation (e.g., ATA, etc.)
        """
        # Calculate the loss
        # import pdb;pdb.set_trace()
        
        ori_loss =  -self.loss(logits, label) if self.targeted else self.loss(logits, label)
        
        if attn_weights_benign is None or attn_weights_adv is None:
            return ori_loss
        else:
            # maximize the difference between the benign and adversarial attention weights
            loss = 0
            for i in range(len(attn_weights_benign)):
                loss += torch.cosine_similarity(attn_weights_benign[i].flatten(), attn_weights_adv[i].flatten(), dim=0)
            # import pdb;pdb.set_trace()
            loss = loss / len(attn_weights_benign)
            return ori_loss - loss