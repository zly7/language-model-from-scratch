import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# We'll start by generating one word
input_ids = tokenizer.encode("I enjoy", return_tensors="pt")
outputs = model(input_ids, use_cache=True)

# We'll take the token with the highest score as our next word
next_token_logits = outputs.logits[:, -1, :]
next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

# We add the new word to our input
input_ids = torch.cat([input_ids, next_token], dim=-1)

# Now we want to generate another word, but we can use the cache to avoid recalculating the attention for "I enjoy"
outputs = model(input_ids, use_cache=True, past_key_values=outputs.past_key_values)

# And we can continue this process for as many words as we want to generate
