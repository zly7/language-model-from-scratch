
from datasets import load_dataset, DatasetDict, load_from_disk
def main():
    print("Loading dataset")
    preprocessed_splits = load_from_disk("wikitext-103-preprocessed-ws-notext-gpt2")
    print("preprocessed_splits train",len(preprocessed_splits["train"][0]["input_ids"])) #这里就是attention mask, input_ids
    from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, GPT2Tokenizer, GPT2Config
    context_length = 128

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # tokenizer = GPT2Tokenizer.from_pretrained("bert-base-uncased")

    config = GPT2Config(n_embd=512, n_layer=8, n_head=8, n_ctx=context_length, 
    bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, vocab_size=len(tokenizer))
    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    from transformers import DataCollatorForLanguageModeling

    # Set the padding token to the EOS token
    tokenizer.pad_token = tokenizer.eos_token

    # Create the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.2) # 这个到这里会报错
    from transformers import Trainer, TrainingArguments
    import datetime
    now = datetime.datetime.now()
    date_string = now.strftime("%m-%d-%H-%M")
    args = TrainingArguments(
        output_dir=f"hug_gpt_train_self/{date_string}/",
        per_device_train_batch_size=8,  # wikitext103 在这个128content_length下，batch_size=8,有10万step
        per_device_eval_batch_size=8,
        eval_steps=5_000,
        logging_steps=2_000,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=3e-3,
        save_steps=5_000,
        fp16=True,
        report_to="tensorboard",
    )
    from trainer import TrainerSelf
    trainer = TrainerSelf(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=None,
        train_dataset=preprocessed_splits["train"],
        eval_dataset=preprocessed_splits["validation"],
    )
    print("Training model starts")
    trainer.train()


if __name__ == "__main__":
    main()

