from transformers import ReformerConfig, ReformerModel,ReformerForMaskedLM,ReformerModelWithLMHead
from transformers import ReformerTokenizer
import torch
# Initialize a Reformer tokenizer
# 下面这句话明明可以下载，服务器不行
# tokenizer = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")

# # Encode some text
# input_text = "Hello, how are you?"
# encoded_text = tokenizer.encode(input_text, return_tensors='pt')

# Initializing a Reformer configuration
device = 'cpu'
batch_size = 4
sequence_length = 1024  # max sequence length
input_ids = torch.randint(high=320, size=(batch_size, sequence_length),device=device)  # 320 is the vocab size
configuration = ReformerConfig(vocab_size=320, num_attention_heads=12,attention_head_size=64, 
             attn_layers=['local','lsh','local','lsh','local','lsh','local','lsh','local','lsh','local',
             'lsh','local','lsh',],
             feed_forward_size=768*4,axial_pos_shape=[32,32],axial_pos_embds_dim=[256,512],hidden_size=768,num_hashes=1,is_decoder=True,num_buckets=32)
print(configuration)

# Initializing a Reformer model (with random weights)
model = ReformerModel(configuration).to(device)

for name,p in model.named_parameters():
    print(name)
    print(p.shape)

out = model(input_ids)
