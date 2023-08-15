# 对应model-gpt2-hug-format

from datasets import load_dataset, DatasetDict, load_from_disk
import torch
import os
import sys
def main():
    root_path = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), ".."), ".."))
    sys.path.append(root_path)
    print("root_path",root_path)
    print("Loading dataset")
    # tokenizer 就是bert的tokenizer，不用gpt
    data_set_path = os.path.join(root_path,"processed_datadir2/bookcorpus-gpt2-2048")
    preprocessed_splits = load_from_disk(data_set_path)
    sequence_length = 2048
    print("train length :", len(preprocessed_splits["train"]))
    from torchscale.architecture.config import RetNetConfig,RetNetConfigDataclass
    from model.model_retnet.retnet_gpt import RetnetGPT


    from transformers import GPT2Tokenizer
    tokenizer_path = os.path.join(root_path,f"tokenizer_save/tokenizer-gpt2-{sequence_length}")
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    print("vocab_size for retnet GPT",str(len(tokenizer)))

    base_config = RetNetConfigDataclass(
        decoder_embed_dim=512,decoder_retention_heads=2,decoder_ffn_embed_dim=1024,decoder_layers=6,activation_fn="gelu",
        dropout=0.0,no_output_layer=False,vocab_size=tokenizer.vocab_size,
    )
    base_medium_config = RetNetConfigDataclass(
        decoder_embed_dim=768,decoder_retention_heads=3,decoder_ffn_embed_dim=768*2,decoder_layers=6,activation_fn="gelu",
        dropout=0.0,no_output_layer=False,vocab_size=tokenizer.vocab_size,
    )
    medium_config = RetNetConfigDataclass(
        decoder_embed_dim=1024,decoder_retention_heads=4,decoder_ffn_embed_dim=2048,decoder_layers=10,activation_fn="gelu",
        dropout=0.0,no_output_layer=False,vocab_size=tokenizer.vocab_size,
    )

    large_config = RetNetConfigDataclass(
        decoder_embed_dim=1024,decoder_retention_heads=4,decoder_ffn_embed_dim=2048,decoder_layers=16,activation_fn="gelu",
        dropout=0.0,no_output_layer=False,vocab_size=tokenizer.vocab_size,
    )

    base_medium_config_permute = RetNetConfigDataclass(
        decoder_embed_dim=768,decoder_retention_heads=3,decoder_ffn_embed_dim=768*2,decoder_layers=6,activation_fn="gelu",
        dropout=0.0,no_output_layer=False,vocab_size=tokenizer.vocab_size,random_permute_times=3,
    )

    base_medium_config_oro = RetNetConfigDataclass(
        decoder_embed_dim=768,decoder_retention_heads=3,decoder_ffn_embed_dim=768*2,decoder_layers=6,activation_fn="gelu",
        dropout=0.0,no_output_layer=False,vocab_size=tokenizer.vocab_size,random_permute_times=-1,
    )
    config = base_medium_config_oro
    model = RetnetGPT(config) #准备搞一个这个当模型，然后这个模型是没有输出MLP的

    from transformers import Trainer,DataCollatorForLanguageModeling
    from TrainArgumentSelf import TrainingArgumentsSelf
    import datetime
    now = datetime.datetime.now()
    date_string = now.strftime("%m-%d-%H-%M")
    from determine_batch_size import get_batch_size
    batch_size = get_batch_size("base","gpt",sequence_length)
    gradient_ac = 12
    if config.decoder_embed_dim == 512:
        batch_size = get_batch_size("base","retnet",sequence_length)
        gradient_ac = 12
    elif config.decoder_embed_dim == 1024:
        batch_size = get_batch_size("large","retnet",sequence_length)
        gradient_ac = 8
    elif config.decoder_embed_dim == 768:
        batch_size = get_batch_size("base","retnet",sequence_length)
        gradient_ac = 12
    else:
        raise ValueError("hidden_size error")
    device_num = torch.cuda.device_count()
    print("one_node_device_num:",device_num)
    Warning("The all gradient ac is about one node,if you use multi node,please decrease the all_gradient_ac")
    assert device_num >= 1
    gradient_ac = gradient_ac // device_num
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    max_steps = 10e5
    args = TrainingArgumentsSelf(
        output_dir=os.path.join(root_path,f"hug_re_pretrain_gpt/bookcorpus_retnet-{date_string}/"),
        per_device_train_batch_size=batch_size,   
        per_device_eval_batch_size=batch_size * 2,
        eval_steps=2000 * gradient_ac,
        logging_steps=20 * gradient_ac,
        gradient_accumulation_steps=gradient_ac,
        max_steps=max_steps,   
        num_train_epochs=20,  
        weight_decay=0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.98,
        adam_epsilon=1e-8,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        learning_rate=1e-4,
        save_steps=1_000 * gradient_ac,
        fp16=False,
        report_to="tensorboard",
        train_audit_probability=0,
        test_step= None,
        per_device_test_batch_size=8,
        all_test_examples_num=128,
        test_dataloader_use_accelerate=True,
        optimizer_type="adamw",
        data_set_path=data_set_path,
        sequence_length=sequence_length,
    )
    from trainer import TrainerSelf
    trainer = TrainerSelf(
        model_name="retnet",
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=preprocessed_splits["train"],
        eval_dataset=preprocessed_splits["validation"],
        test_dataset=preprocessed_splits.get("test",preprocessed_splits["validation"]),
    )
    print("Training model starts")
    trainer.train()


if __name__ == "__main__":
    main()


