import torch.nn as nn


from dataclasses import dataclass,field

from .embedding import BERTEmbedding
from ..backbone.modeling_outputs import MaskedLMOutput,BaseModelOutputWithPastAndCrossAttentions
from ..backbone.bb_attention import CosformerAttention,BidirectionalSelfAttention,DirectMultiplyAttentionNotRight
from ..backbone.bb_basic import LayerNorm, new_gelu, MLP
import torch
from ..backbone.bb_config import LMconfig
@dataclass
class BertConfig(LMconfig):
    use_directmul:bool = field(default=False)
    pass


class BidirectionalBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x,attn_mask=None):
        x = x + self.attn(self.ln_1(x),attn_mask = attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class cosFormerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CosformerAttention(embed_dim=config.n_embd, num_heads=config.n_head, dropout_rate=config.dropout,causal=False,batch_first=True)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x,attn_mask=None):
        x = x + self.attn(self.ln_1(x),attn_mask = attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class directMulBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = DirectMultiplyAttentionNotRight(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x,attn_mask=None):
        x = x + self.attn(self.ln_1(x),attn_mask = attn_mask).x
        x = x + self.mlp(self.ln_2(x))
        return x


class BERTEncoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, config:BertConfig):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=config.vocab_size, embed_size=config.n_embd)

        # multi-layers transformer blocks, deep network
        if config.use_cosformer:
            self.transformer_blocks = nn.ModuleList(
                [cosFormerBlock(config) for _ in range(config.n_layer)])
        elif config.use_directmul:
            self.transformer_blocks = nn.ModuleList(
                [directMulBlock(config) for _ in range(config.n_layer)])
        else:
            self.transformer_blocks = nn.ModuleList(
                [BidirectionalBlock(config) for _ in range(config.n_layer)])

    def forward(self, x, segment_info,attn_mask=None):
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, attn_mask = attn_mask)

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=x)





class BertLM(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, config:BertConfig):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.config = config
        self.bert = BERTEncoder(config)
        
        self.linear = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x,labels=None,token_type_ids=None,attn_mask = None):
        basic_output = self.bert(x,segment_info=token_type_ids,attn_mask=attn_mask)
        logits = self.linear(basic_output.last_hidden_state)
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1,logits.size(-1)),labels.view(-1))
        else:
            loss = None
        return MaskedLMOutput(loss=loss,logits=logits)
    
    def save_pretrained(self, save_directory):
        import os
        print(f"saving checkpoint to {save_directory}")
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, 'pretrain_weight.pt'))
    
    # @classmethod
    # def from_pretrained(cls, override_args=None,model_hf=None):
        # override_args = override_args or {} # default to empty dict
        # # only dropout can be overridden see more notes below
        # assert all(k == 'dropout' for k in override_args)
        # from transformers import BertForMaskedLM

        # # n_layer, n_head and n_embd are determined from model_type
        # config_args = {
        #     'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        #     'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        #     'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        #     'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        # }[model_type]
        # print("forcing vocab_size=50257, block_size=1024, bias=True")
        # config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        # config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # config_args['bias'] = True # always True for GPT model checkpoints
        # # we can override the dropout rate, if desired
        # if 'dropout' in override_args:
        #     print(f"overriding dropout rate to {override_args['dropout']}")
        #     config_args['dropout'] = override_args['dropout']
        # # create a from-scratch initialized minGPT model
        # config = GPTConfig(**config_args)
        # model = GPT(config)
        # sd = model.state_dict()
        # sd_keys = sd.keys()
        # sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # # init a huggingface/transformers model
        # if model_hf is None:
        #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        # sd_hf = model_hf.state_dict()

        # # copy while ensuring all of the parameters are aligned and match in names and shapes
        # sd_keys_hf = sd_hf.keys()
        # sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        # sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        # transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # # this means that we have to transpose these weights when we import them
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # for k in sd_keys_hf:
        #     if any(k.endswith(w) for w in transposed):
        #         # special treatment for the Conv1D weights we need to transpose
        #         assert sd_hf[k].shape[::-1] == sd[k].shape
        #         with torch.no_grad():
        #             sd[k].copy_(sd_hf[k].t())
        #     else:
        #         # vanilla copy over the other parameters
        #         assert sd_hf[k].shape == sd[k].shape
        #         with torch.no_grad():
        #             sd[k].copy_(sd_hf[k])

        # return model
    


