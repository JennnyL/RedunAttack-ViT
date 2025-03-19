
import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from timm.models.vision_transformer import Attention,Mlp, Block
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List


import torch

from ..utils import *
from ..attack import Attack


attn_weights = []

q_rest = {}
k_rest = {}
v_rest = {}

def Wrapped_Attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    
    named_id = self.named_id
    global q_rest, k_rest, v_rest
    num_tokens = q.shape[2]
    filling = False
    if named_id in q_rest:
        # concatenate the q, k, v
        filling = True
        q = torch.cat([q, q_rest[named_id]], dim=1)
        k = torch.cat([k, k_rest[named_id]], dim=1)
        v = torch.cat([v, v_rest[named_id]], dim=1)
    else:
        # randomly sample a subset of tokens (10%)
        sample_num_tokens = int(0.1 * num_tokens)
        num_heads = q.shape[1]
        selected_token_ids = [torch.from_numpy(np.random.choice(torch.arange(1,num_tokens), sample_num_tokens,replace=False)) for _ in range(num_heads)]
        selected_token_ids = torch.stack(selected_token_ids, dim=0)
        # selected_token_ids shape: (head, sample_num_tokens)
        # q shape: (B, num_heads, num_tokens, head_dim)
        # fetch the sampled tokens
        import pdb;pdb.set_trace()
        q_rest[named_id] = q.gather(2, selected_token_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        k_rest[named_id] = k.gather(2, selected_token_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        v_rest[named_id] = v.gather(2, selected_token_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        
    
    
    q, k = self.q_norm(q), self.k_norm(k)
    # import pdb;pdb.set_trace()
    
    
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    
    # attn: (N, num_heads, P, P)
    # global attn_weights
    # attn_weights.append(attn)
    # import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()
    # random drop 50% of the attention weights
    # attn = attn * (torch.rand_like(attn) > 0.5).float()
    x = attn @ v
    
    if filling:
        x = x[:, :num_tokens]
    
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
 
N = 3    
    
def Wrapper_FFN_forward(self, input):
    output = 0.
    global N
    current_N = np.random.randint(1, N+1)
    for n in range(current_N):
        x = self.fc1(input)
        x = self.act(x)
        x = x * (torch.rand_like(x)>0.3).float()
        # x = self.drop1(x)
        x = self.fc2(x)
        # x = self.drop2(x)
        output += x
    output = output / current_N
    return output


class RESTAttack(Attack):    
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
    
    
    
    
    
    
    def wrap_attention(self):
        # transform the attention module in the model to the wrapped attention module
        module_id = 0
        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                module.named_id = module_id
                module.forward = Wrapped_Attention_forward.__get__(module)
            # if isinstance(module, Mlp):
            #     module.forward = Wrapper_FFN_forward.__get__(module)
        



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
            # attn_weights_adv = attn_weights
            # attn_weights = []
            
            # import pdb;pdb.set_trace()
            loss = self.get_loss(logits, label, attn_weights_benign)
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