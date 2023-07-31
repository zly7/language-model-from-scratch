from transformers import DataCollatorForLanguageModeling
import random

class MyDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15):
        super().__init__(tokenizer, mlm_probability)

    def mask_whole_word(self, inputs, mlm_probability=0.15):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "Remove the --mlm flag if you want to use this tokenizer."
            )

        inputs = inputs.clone()
        probability_matrix = torch.full(inputs.shape, mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # Set the probability of masked tokens to 0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs[~masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # Prepare the token_ids for masked tokens
        masked_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        inputs[masked_indices] = masked_token_ids

        return inputs

# Example usage:
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

data_collator = MyDataCollatorForLanguageModeling(tokenizer)

# Some example input text
text = "Hello, my name is ChatGPT. I'm here to help you with your questions."

# Tokenize the text
inputs = tokenizer(text, return_tensors='pt')

# Use the custom mask function
inputs['input_ids'] = data_collator.mask_whole_word(inputs['input_ids'])

# Output the modified text
print(tokenizer.decode(inputs['input_ids'][0]))
