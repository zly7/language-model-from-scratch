from transformers import ReformerConfig, ReformerModel

# Initializing a Reformer configuration
configuration = ReformerConfig()
print(configuration)

# Initializing a Reformer model (with random weights)
model = ReformerModel(configuration)

for name,p in model.named_parameters():
    print(name)
    print(p.shape)

# Accessing the model configuration
configuration = model.config