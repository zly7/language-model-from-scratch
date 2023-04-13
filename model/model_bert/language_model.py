import torch.nn as nn


from dataclasses import dataclass

from .embedding import BERTEmbedding
from ..backbone.modeling_outputs import MaskedLMOutput,BaseModelOutputWithPastAndCrossAttentions
from ..backbone.bb_attention import CosformerAttention,BidirectionalSelfAttention
from ..backbone.bb_basic import LayerNorm, new_gelu, MLP
import torch
@dataclass
class BertConfig:
    block_size: int = 1024
    vocab_size: int = 50304 
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias = False
    use_cosformer: bool = False


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
    


