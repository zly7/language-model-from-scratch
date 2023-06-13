from dataclasses import dataclass,field
@dataclass
class LMconfig():
    block_size: int = field(default=1024)
    vocab_size: int = field(default=50304)
    n_layer: int = field(default=12)
    n_head: int = field(default=12)
    n_embd: int = field(default=768)
    dropout: float = field(default=0.0)
    bias: bool = field(default=False)
    use_cosformer: bool = field(default=False)
    use_SDPA : bool = field(default=False)
    use_reformer: bool = field(default=False)
