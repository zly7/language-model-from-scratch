
import math
import inspect
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from ..backbone.modeling_outputs import CausalLMOutputWithCrossAttentions,BaseBlockOutput

from ..backbone.bb_basic import LayerNorm, new_gelu, MLP
from ..backbone.bb_attention import CausalSelfAttention,CosformerAttention
import math
from ..backbone.bb_config import LMconfig
from torchscale.architecture.config import RetNetConfig,RetNetConfigDataclass
from torchscale.architecture.retnet import RetNetDecoder

class RetnetGPT(nn.Module):

    def __init__(self, config : Union[RetNetConfig , RetNetConfigDataclass]):
        super().__init__()
       

        # Input embedding layer
        self.input_emb = nn.Embedding(config.vocab_size, config.decoder_embed_dim)
        
        # RetNetDecoder
        self.decoder = RetNetDecoder(config,embed_tokens=self.input_emb,output_projection=None) # 这里是带了输出层的
        
        if config.no_output_layer is True:
            raise ValueError("You should set a no_output_layer")

        self.config = config

    def forward(self, idx, targets=None, autoShiftLeftAndTrain=True):
        if autoShiftLeftAndTrain is True:
            # zly: the target should shift left,(1,0) means the left padding position is 1, don't do right padding
            # targets =  torch.nn.functional.pad(idx, (1, 0), mode='constant', value=-100)[:, :-1]

            targets =  (torch.nn.functional.pad(idx, (0, 1), mode='constant', value=-100)[:, 1:]).contiguous()



        logits,tuple_other = self.decoder(prev_output_tokens = idx) # shape (b, t, c)
        # torch.cuda.synchronize(0)

        if targets is not None:
            # if we are given some desired targets also calculate the loss  
            # zly: softmax is not needed here, because F.cross_entropy() will do it
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        else:
            loss = None
        preds = torch.argmax(logits, dim=-1)
        active_loss_position = (targets != -100)  # There use muktiplier also work
        acc = (preds[active_loss_position] == targets[active_loss_position]).float().mean() if targets is not None else None
        top_k_acc = None
        # if targets is not None:
        #     top_k = 5
        #     torch.cuda.synchronize(0)
        #     top_k_preds = torch.topk(logits, k=top_k, dim=-1).indices
        #     top_k_correct = torch.eq(top_k_preds, targets.unsqueeze(-1)).any(dim=-1).float().sum()
        #     top_k_acc = top_k_correct / active_loss_position.nonzero().size(0)
        # else:
        #     top_k_acc = None
        
        if hasattr(self.config, 'visualize') and self.config.visualize:
            return CausalLMOutputWithCrossAttentions(
                logits=logits,
                loss=loss,
                accuracy=acc,
                topkaccuracy=top_k_acc,
            )
        else:
            return CausalLMOutputWithCrossAttentions(
                logits=logits,
                loss=loss,
                accuracy=acc,
                topkaccuracy=top_k_acc,
            )
        

    def save_pretrained(self, save_directory):
        import os
        import json
        print(f"saving checkpoint to {save_directory}")
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, 'pretrain_weight.pt'))
        with open(os.path.join(save_directory, 'model_hyperparameter.json'), "w") as f:
            json.dump(self.config.__dict__, f, indent=4)