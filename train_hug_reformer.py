# 对应model-gpt2-hug-format

from datasets import load_dataset, DatasetDict, load_from_disk
import torch
from determine_batch_size import get_batch_size
def main():
    print("Loading dataset")
    preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-gpt2-story-train512-test256")
    print("train length :", len(preprocessed_splits["train"]))
    sequence_length = 1024
    from transformers import ReformerTokenizer
    tokenizer = ReformerTokenizer.from_pretrained(f"./tokenizer_save/tokenizer-gpt2-{sequence_length}")
    tokenizer.eos_token_id = tokenizer.pad_token_id
    from transformers import ReformerConfig,DataCollatorForLanguageModeling
    config = ReformerConfig(vocab_size=tokenizer.vocab_size, n_embd=768, 
                n_layer=12, n_head=12)
 
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel(config)
    # model.load_state_dict(torch.load("./hug_gpt_train_self/03-30-17-03/checkpoint-20001/pretrain_weight.pt"))
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
        output_dir=f"hug_gpt_pretrain/{date_string}/",
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
        warmup_steps= 1000,
        lr_scheduler_type="cosine",
        # lr_scheduler_type="constant",
        # learning_rate=1e-4,
        learning_rate= 5e-4,
        save_steps=1_000 * gradient_ac,
        fp16=True,
        report_to="tensorboard",
        train_audit_probability=0,
        test_step=5000*gradient_ac,
        per_device_test_batch_size=32,
        all_test_examples_num=0,
        test_dataloader_use_accelerate=True,
        optimizer_type="adamw",
    )
    from trainer import TrainerSelf
    trainer = TrainerSelf(
        model_name="huggingface gpt",
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


