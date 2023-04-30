# 对应model-gpt2-hug-format

from datasets import load_dataset, DatasetDict, load_from_disk
import torch
def main():
    print("Loading dataset")
    # preprocessed_splits = load_from_disk("wikitext-103-preprocessed-ws-notext-gpt2-128-wtest-v2")
    # print("train length :", len(preprocessed_splits["train"]))
    # print("preprocessed_splits train : ", len(preprocessed_splits["train"][0]["input_ids"]))
    # print("preprocessed_splits test : ", len(preprocessed_splits["test"][0]))
    preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-prepro-gpt2-512-wo-train-for-visualize")
    from model.model_gpt.model_gpt2_hug_formet import GPT, GPTConfig
    from transformers import GPT2LMHeadModel,GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer_save/tokenizer-gpt2-512")
    print("vocab_size for GPT2",str(len(tokenizer)))
    model = GPT2LMHeadModel.from_pretrained("./model_save/gpt2-finetuned-wikitext103")
    model.config.visualize = True



    from TrainArgumentSelf import TrainingArgumentsSelf
    import datetime
    now = datetime.datetime.now()
    date_string = now.strftime("%m-%d-%H-%M")
    gradient_ac = 10
    max_steps = 13000*5 * gradient_ac
    args = TrainingArgumentsSelf(
        output_dir=f"vanilla_gpt_pretrain/{date_string}/",
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
        fp16=True,
        report_to="tensorboard",
        train_audit_probability=0,
        test_step=10000*gradient_ac,
        per_device_test_batch_size=8,
        all_test_examples_num=256,
        test_dataloader_use_accelerate=True,
        
    )
    from trainer import TrainerSelf
    trainer = TrainerSelf(
        model_name="huggingface gpt visualize",
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=None,
        train_dataset=preprocessed_splits["validation"],
        eval_dataset=preprocessed_splits["validation"],
        test_dataset=preprocessed_splits["validation"]
    )
    print("Training model starts")
    # trainer.evaluate(0)
    trainer.test(0)


if __name__ == "__main__":
    main()


