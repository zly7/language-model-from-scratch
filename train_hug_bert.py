
from datasets import load_dataset, DatasetDict, load_from_disk
from model.model_bert.language_model import  BertConfig,BertLM
# 这句话访问不到是没有访问到module,需要在__init__.py里面引入，可以想象成__init__.py是本题
# from model.model_bert.bert import BERTs
# from model.model_bert.language_model import BERTLM
from transformers import AutoTokenizer
import torch
def main():
    print("Loading dataset")
    preprocessed_splits = load_from_disk("wikitext-103-bert-512-without-test")
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer_save/tokenizer-bert-base-uncased-512")
    print("tokenizer:",str(tokenizer))

    # Create the model
    config = BertConfig(vocab_size=tokenizer.vocab_size, n_embd=768, 
                n_layers=12, n_head=12, dropout=0.1, use_cosformer=False)
    LMmodel = BertLM(config)
    model_size = sum(t.numel() for t in LMmodel.parameters())
    print(f"Bert size: {model_size/1000**2:.1f}M parameters")
    from transformers import DataCollatorForLanguageModeling

    LMmodel.load_state_dict(torch.load("./hug_bert_train_self/04-05-11-55/checkpoint-13576/pretrain_weight.pt"))

    # Create the data collator
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    from TrainArgumentSelf import TrainingArgumentsSelf
    import datetime
    now = datetime.datetime.now()
    date_string = now.strftime("%m-%d-%H-%M")
    args = TrainingArgumentsSelf(
        output_dir=f"hug_bert_train_self/{date_string}/",
        per_device_train_batch_size=16,   # 差不多13574个batch
        per_device_eval_batch_size=32,
        eval_steps=5_000,
        logging_steps=200,
        gradient_accumulation_steps=1,
        max_steps=13000*5,  
        num_train_epochs=20,  # 一个epoch差不多是1H，2GPU
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=3e-3,
        save_steps=5_000,
        fp16=True,
        report_to="tensorboard",
        train_audit_probability=0,
        test_step=5000,
        per_device_test_batch_size=8,
        all_test_examples_num=256,
        test_dataloader_use_accelerate=True,
    )
    from trainer import TrainerSelf
    trainer = TrainerSelf(
        model_name="self bert",
        model=LMmodel,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=preprocessed_splits["train"],
        eval_dataset=preprocessed_splits["validation"],
        test_dataset=preprocessed_splits["validation"]
    )
    print("Training model starts")
    trainer.train()


if __name__ == "__main__":
    main()

