
import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from timm.models.vision_transformer import Attention,Mlp, Block, checkpoint_seq, VisionTransformer
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
        append_token = torch.einsum('tn,bnd->btd', opt_tokens, x[:,1:])

        x = torch.cat([x, append_token], dim=1)
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
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., resize_rate=1.1, diversity_prob=0.5, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='GI-FGSM',  s=10, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device, **kwargs)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.s = s
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.pre_epoch = 3
        self.num_tokens = 10
        self.token_dim = 768
        self.num_patches = 196
        self.model = self.wrap_forward_features(self.model)

    def init_robust_delta(self):
        delta = torch.rand(self.num_tokens, self.num_patches).to(self.device)
        delta.requires_grad = True
        return delta
    
    def update_robust_delta(self, delta, grad, alpha, **kwargs):
        # grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
        # scaled_grad = grad # / (grad_norm + 1e-20)
        delta = delta - grad.sign() * alpha
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
        raise Exception('The model does not contain VisionTransformer module')
        



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
        
        robust_delta = self.init_robust_delta()
        global opt_tokens
        opt_tokens = robust_delta
        
        
    
        
        momentum = 0.
        momentum_robust = 0.
        delta = self.init_delta(data).to(self.device)
        for _ in range(self.epoch):
            # Obtain the output
            
            opt_tokens = robust_delta
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            # Calculate the loss
            loss = self.get_loss(logits, label)
            with torch.no_grad():
                opt_tokens = None
                logits_robust = self.get_logits(self.transform(data+delta, momentum=momentum_robust))
                pre_loss = self.get_loss(logits_robust, label)
            # print(loss.item(), pre_loss.item())
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            
            grad_opt = self.get_grad(loss, robust_delta)
            
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            momentum_robust = self.get_robust_momentum(grad_opt, momentum_robust)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
            # robust_delta = self.update_robust_delta(robust_delta, momentum_robust, self.alpha/10)
        
        # import pdb;pdb.set_trace()
        
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