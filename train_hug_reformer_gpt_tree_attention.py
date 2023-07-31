# 对应huggingface 版本的reformer，现在先尝试跑一下，看这个的训练效果

from datasets import load_dataset, DatasetDict, load_from_disk
import torch
from determine_batch_size import get_batch_size
from compute_metrics import compute_metrics_for_masklm
def main():
    print("Loading dataset")
    data_set_path = "./processed_datadir2/wikitext-103-story-bert-2048"
    preprocessed_splits = load_from_disk(data_set_path)
    print("train length :", len(preprocessed_splits["train"]))
    sequence_length = 2048
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"./tokenizer_save/tokenizer-bert-base-uncased-{sequence_length}")
    tokenizer.eos_token_id = tokenizer.pad_token_id
    from transformers import ReformerConfig,DataCollatorForLanguageModeling
    # config = ReformerConfig(vocab_size=tokenizer.vocab_size, num_attention_heads=12,attention_head_size=64, 
    #          #attn_layers=['local','lsh','local','lsh','local','lsh','local','lsh','local','lsh','local','lsh','local','lsh',],
    #          attn_layers=['lsh'],
    #          feed_forward_size=768*4,axial_pos_shape=[32,32],axial_pos_embds_dim=[256,512],hidden_size=768,num_buckets=32,
    #          whether_use_tree_attention=True,num_hashes=4)
    config = ReformerConfig(vocab_size=tokenizer.vocab_size, num_attention_heads=8,attention_head_size=128, 
             attn_layers=['local','local','lsh','local','local','local','lsh','local','local','local','lsh','local','local','local','lsh','local',],
            #  attn_layers=['lsh',],
             feed_forward_size=1024*4,axial_pos_shape=[32,64],axial_pos_embds_dim=[256,768],hidden_size=1024,num_buckets=32,
             whether_use_tree_attention=True,num_hashes=4,is_decoder=True)
 
    from transformers import ReformerModelWithLMHead,ReformerForMaskedLM
    model = ReformerModelWithLMHead(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"gpt-reformer-tree-attention size: {model_size/1000**2:.1f}M parameters")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    from TrainArgumentSelf import TrainingArgumentsSelf
    import datetime
    now = datetime.datetime.now()
    date_string = now.strftime("%m-%d-%H-%M")

    if config.hidden_size == 1024:
        batch_size = get_batch_size("large","reformer gpt",sequence_length)
        gradient_ac = 8
    elif config.hidden_size == 768:
        batch_size = get_batch_size("base","reformer gpt",sequence_length)
        gradient_ac = 12
    else:
        raise ValueError("hidden_size error")
    max_steps = 10e5  
    args = TrainingArgumentsSelf(
        output_dir=f"hug_re_pretrain_gpt/new_data_tree-{date_string}/",
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
        learning_rate= 1e-4,
        save_steps=1_000 * gradient_ac,
        fp16=False,
        report_to="tensorboard",
        train_audit_probability=0,
        test_step=5000*gradient_ac,
        per_device_test_batch_size=32,
        all_test_examples_num=0,
        test_dataloader_use_accelerate=True,
        optimizer_type="adamw",
        sequence_length=sequence_length,
        data_set_path=data_set_path,
    )
    from trainer import TrainerSelf
    trainer = TrainerSelf(
        model_name="huggingface reformer",
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=preprocessed_splits["train"],
        eval_dataset=preprocessed_splits["validation"],
        test_dataset=preprocessed_splits["test"],
        compute_metrics=compute_metrics_for_masklm,
    )
    print("Training model starts test for weight decay parameter")
    trainer.train()


if __name__ == "__main__":
    main()


