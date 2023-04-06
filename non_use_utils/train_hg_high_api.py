
from datasets import load_dataset, DatasetDict, load_from_disk

# if __name__ == "main":
print("Loading dataset")
preprocessed_splits = load_from_disk("wikitext-103-preprocessed")
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, GPT2Tokenizer, GPT2Config
context_length = 128
# Initialize tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# config = AutoConfig.from_pretrained(
#     "gpt2",
#     vocab_size=len(tokenizer),
#     n_ctx=context_length,
#     bos_token_id=tokenizer.bos_token_id, # 原本是开始和结束的特殊字符
#     eos_token_id=tokenizer.eos_token_id,
# )
config = GPT2Config(n_embd=512, n_layer=8, n_head=8, n_ctx=context_length, 
bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, vocab_size=len(tokenizer))
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
from transformers import DataCollatorForLanguageModeling

# Set the padding token to the EOS token
tokenizer.pad_token = tokenizer.eos_token

# 这个trainer 一直要登陆

# Create the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
from transformers import Trainer, TrainingArguments

model.generate()

args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=3e-2,
    save_steps=5_000,
    fp16=True,
    report_to="tensorboard",
    # push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=preprocessed_splits["train"],
    eval_dataset=preprocessed_splits["validation"],
)
print("Training model starts")
trainer.train()

