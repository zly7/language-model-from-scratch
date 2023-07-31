# 对应huggingface 版本的reformer，现在先尝试跑一下，看这个的训练效果,主要是换成这种char token的方式

from datasets import load_dataset, DatasetDict, load_from_disk
import torch
from determine_batch_size import get_batch_size
def main():
    print("Loading dataset")
    data_set_path = "./processed_datadir2/wikitext-103-story-chartoken-bert-2048"
    preprocessed_splits = load_from_disk(data_set_path)
    print("train length :", len(preprocessed_splits["train"]))
    sequence_length = 2048
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"./tokenizer_save/byt5-tokenizer")
    tokenizer.eos_token_id = tokenizer.pad_token_id
    from model.model_gpt.model_gpt2_hug_formet import GPT, GPTConfig
    config = GPTConfig(vocab_size=tokenizer.vocab_size, n_embd=768, 
                n_layer=12, n_head=12, dropout=0.1, use_cosformer=False,block_size=sequence_length)

    model = GPT(config)
    from transformers import DataCollatorForLanguageModeling
    model_size = sum(t.numel() for t in model.parameters())
    print(f"vanilla gpt size: {model_size/1000**2:.1f}M parameters")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    from TrainArgumentSelf import TrainingArgumentsSelf
    import datetime
    now = datetime.datetime.now()
    date_string = now.strftime("%m-%d-%H-%M")
    gradient_ac = 12
    batch_size = get_batch_size("base","gpt",sequence_length)
    
    # max_steps = 13000*5 * gradient_ac
    max_steps = 5e5
    args = TrainingArgumentsSelf(
        output_dir=f"hug_re_pretrain_chartoken_gpt/vanilla-{date_string}/",
        per_device_train_batch_size=batch_size,   # 16的时候，训练只消耗17.5G显存,24bacth消耗23G,不使用混合精度训练反而24batch还没法用了， 
        per_device_eval_batch_size=batch_size * 2,
        eval_steps=1000 * gradient_ac,
        logging_steps=20 * gradient_ac,
        gradient_accumulation_steps=gradient_ac,
        max_steps=max_steps,   
        num_train_epochs=20,  
        weight_decay=0.01,
        # weight_decay = 0.1,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        # adam_epsilon = 1e-6,
        adam_epsilon= 1e-8,
        warmup_steps= 1000,  # 这里还是没明白为什么warmup_steps 最后表现出来是4000
        lr_scheduler_type="cosine",
        # lr_scheduler_type="constant",
        # learning_rate=1e-4,
        learning_rate= 1e-4,
        save_steps=1_000 * gradient_ac,
        fp16=False,
        report_to="tensorboard",
        train_audit_probability=0,
        test_step=None,
        per_device_test_batch_size=32,
        all_test_examples_num=0,
        test_dataloader_use_accelerate=True,
        optimizer_type="adamw",
        sequence_length=sequence_length,
    )
    from trainer import TrainerSelf
    trainer = TrainerSelf(
        model_name="self gpt",
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=preprocessed_splits["train"],
        eval_dataset=preprocessed_splits["validation"],
        test_dataset=preprocessed_splits["test"]
    )
    print("Training model starts test for weight decay parameter")
    trainer.train()


if __name__ == "__main__":
    main()


