# 对应model-gpt2-hug-format

from datasets import load_dataset, DatasetDict, load_from_disk
import torch
def main():
    print("Loading dataset")
    preprocessed_splits = load_from_disk("./processed_datatdir/wikitext-103-preprocessed-ws-notext-gpt2-128-wtest-v2")
    print("train length :", len(preprocessed_splits["train"]))
    # print("preprocessed_splits train : ", len(preprocessed_splits["train"][0]["input_ids"]))
    # print("preprocessed_splits test : ", len(preprocessed_splits["test"][0]))
    from model.model_gpt.model_gpt2_hug_formet import GPT, GPTConfig
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer_save/tokenizer-gpt2-128")
    print("vocab_size for GPT2",str(len(tokenizer)))
    config = GPTConfig(n_embd=512, n_layer=8, n_head=8, block_size=128,bias=False,vocab_size=len(tokenizer),use_cosformer=True)

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
    gradient_ac = 10
    max_steps = 13000*5 * gradient_ac
    args = TrainingArgumentsSelf(
        output_dir=f"cos_gpt_pretrain/{date_string}/",
        per_device_train_batch_size=20,   # 16的时候，训练只消耗17.5G显存,24bacth消耗23G,不使用混合精度训练反而24batch还没法用了， 
        per_device_eval_batch_size=32,
        eval_steps=1000 * gradient_ac,
        logging_steps=20 * gradient_ac,
        gradient_accumulation_steps=gradient_ac,
        max_steps=max_steps,   
        num_train_epochs=20,  # 一个epoch差不多是1H，2GPU
        weight_decay=0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-6,
        warmup_steps=200 * gradient_ac,
        lr_scheduler_type="cosine",
        learning_rate=3e-4,
        save_steps=1_000 * gradient_ac,
        fp16=False,
        report_to="tensorboard",
        train_audit_probability=0,
        test_step=10000*gradient_ac,
        per_device_test_batch_size=8,
        all_test_examples_num=256,
        test_dataloader_use_accelerate=True,
        
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


