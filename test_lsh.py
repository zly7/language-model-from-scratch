# should fit in ~ 5gb - 8k embeddings

import torch
from model.backbone.bb_attention import LSHAttention

model = LSHAttention().cuda()
qk = torch.randn(1, 8192, 512).cuda()
v = torch.randn(1, 8192, 512).cuda()
y = model(qk,v)