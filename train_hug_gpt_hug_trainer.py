# 对应model-gpt2-hug-format

from datasets import load_dataset, DatasetDict, load_from_disk
import torch
def main():
    print("Loading dataset")
    preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-gpt2-story-train512-test256")
    print("train length :", len(preprocessed_splits["train"]))
    
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer_save/tokenizer-gpt2-512")
    tokenizer.eos_token_id = tokenizer.pad_token_id
    from transformers import GPT2Config
    config = GPT2Config(vocab_size=tokenizer.vocab_size, n_embd=768, 
                n_layer=12, n_head=12)
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel(config)
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # model.load_state_dict(torch.load("./hug_gpt_train_self/03-30-17-03/checkpoint-20001/pretrain_weight.pt"))

    from TrainArgumentSelf import TrainingArgumentsSelf
    import datetime
    now = datetime.datetime.now()
    date_string = now.strftime("%m-%d-%H-%M")
    gradient_ac = 10
    max_steps = 13000*5 * gradient_ac
    from transformers import TrainingArguments,Trainer



  
    # print("Training model starts")
    args = TrainingArguments(
        output_dir=f"hug_gpt_pretrain_hug_trainer/{date_string}/",
        per_device_train_batch_size=12,   # 16的时候，训练只消耗17.5G显存,24bacth消耗23G,不使用混合精度训练反而24batch还没法用了， 
        per_device_eval_batch_size=24,
        eval_steps=1000 * gradient_ac,
        logging_steps=20 * gradient_ac,
        gradient_accumulation_steps=gradient_ac,
        evaluation_strategy="steps",
        max_steps=max_steps,   
        num_train_epochs=20,  # 一个epoch差不多是1H，2GPU
        weight_decay=0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-6,
        warmup_steps=200 * gradient_ac,
        # lr_scheduler_type="cosine",
        lr_scheduler_type="fixed",
        learning_rate=1e-4,
        save_steps=1_000 * gradient_ac,
        fp16=True,
        report_to="tensorboard",
    )
    trainer = Trainer(model=model, args=args, train_dataset=preprocessed_splits["train"], 
            eval_dataset=preprocessed_splits["validation"], tokenizer=tokenizer,data_collator=data_collator,compute_metrics=None)
    trainer.train()


if __name__ == "__main__":
    main()


