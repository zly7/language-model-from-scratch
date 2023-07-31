
from datasets import load_dataset, DatasetDict, load_from_disk
from model.model_bert.language_model import  BertConfig,BertLM
# 这句话访问不到是没有访问到module,需要在__init__.py里面引入，可以想象成__init__.py是本体
# from model.model_bert.bert import BERTs
# from model.model_bert.language_model import BERTLM
from transformers import AutoTokenizer
import torch
from determine_batch_size import get_batch_size
assert torch.__version__ >= "2.0.0"
def main():
    print("Loading dataset")
    # preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-preprocessed-ws-notext-bert-128-wtest")
    # preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-story-bert-1024")
    preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-story-bert-512")
    sequence_length = 512
    tokenizer = AutoTokenizer.from_pretrained(f"./tokenizer_save/tokenizer-bert-base-uncased-{sequence_length}")
    print("tokenizer:",str(tokenizer))

    # Create the model
    config = BertConfig(vocab_size=tokenizer.vocab_size, n_embd=768, 
                n_layer=12, n_head=12, dropout=0.1, use_cosformer=False,use_SDPA=True)
    LMmodel = BertLM(config)
    model_size = sum(t.numel() for t in LMmodel.parameters())
    print(f"Bert size: {model_size/1000**2:.1f}M parameters")
    for name, param in LMmodel.named_parameters():
        print(name, param.shape)
    from transformers import DataCollatorForLanguageModeling

    # LMmodel.load_state_dict(torch.load("./hug_bert_train_self/04-05-11-55/checkpoint-13576/pretrain_weight.pt"))

    # Create the data collator
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    from TrainArgumentSelf import TrainingArgumentsSelf
    import datetime
    now = datetime.datetime.now()
    date_string = now.strftime("%m-%d-%H-%M")
    gradient_ac = 4
    # max_steps = 13000*5 * gradient_ac
    max_steps = 5e3
    batch_size = get_batch_size("base","bert",sequence_length)
    args = TrainingArgumentsSelf(
        # output_dir=f"vanilla_bert_pretrain/{date_string}/",
        output_dir=f"speed_test/torch2_fastattention_bert/sequence{sequence_length}/",
        per_device_train_batch_size=batch_size,   # 16的时候，训练只消耗17.5G显存,24bacth消耗23G,不使用混合精度训练反而24batch还没法用了， 
        per_device_eval_batch_size=batch_size,
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
        learning_rate=1e-3,
        save_steps=1_000 * gradient_ac,
        fp16=True,
        report_to="tensorboard",
        train_audit_probability=0,
        # test_step=5000*gradient_ac,
        test_step=None,
        optimizer_type="sgd",
        sgd_momentum=0.1,
        sequence_length=sequence_length,

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

