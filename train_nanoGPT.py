# 对应model-gpt2-hug-format

from datasets import load_dataset, DatasetDict, load_from_disk
import torch
def main():
    print("Loading dataset")
    preprocessed_splits = load_from_disk("wikitext-103-preprocessed-ws-notext-gpt2-128-wtest-v2")
    print("train length :", len(preprocessed_splits["train"]))
    # print("preprocessed_splits train : ", len(preprocessed_splits["train"][0]["input_ids"]))
    # print("preprocessed_splits test : ", len(preprocessed_splits["test"][0]))
    from model.model_gpt.model_gpt2_hug_formet import GPT, GPTConfig
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer_save/tokenizer-gpt2-128")
    print("vocab_size for GPT2",str(len(tokenizer)))
    config = GPTConfig(n_embd=512, n_layer=8, n_head=8, block_size=128,bias=False,vocab_size=len(tokenizer))

    model = GPT(config)
    # model.load_state_dict(torch.load("./hug_gpt_train_self/03-30-17-03/checkpoint-20001/pretrain_weight.pt"))

    from transformers import DataCollatorForLanguageModeling


    # Create the data collator
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False) # 用这个反而肯定要pad
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.2) # 这个到这里会报错
    from transformers import Trainer
    from TrainArgumentSelf import TrainingArgumentsSelf
    import datetime
    now = datetime.datetime.now()
    date_string = now.strftime("%m-%d-%H-%M")
    args = TrainingArgumentsSelf(
        output_dir=f"hug_gpt_train_self/{date_string}/",
        per_device_train_batch_size=8,  # wikitext103 在这个128content_length下，batch_size=8,有10万step
        per_device_eval_batch_size=16,
        eval_steps=5_000,
        logging_steps=500,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=3e-3,
        save_steps=5_000,
        fp16=True,
        report_to="tensorboard",
        test_step=5000,
        per_device_test_batch_size=32,
        all_test_examples_num=256,
        train_audit_probability=0.0003,
    )
    from trainer import TrainerSelf
    trainer = TrainerSelf(
        model_name="self gpt",
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=None,
        train_dataset=preprocessed_splits["train"],
        eval_dataset=preprocessed_splits["validation"],
        test_dataset=preprocessed_splits["test"]
    )
    print("Training model starts")
    trainer.train()


if __name__ == "__main__":
    main()


