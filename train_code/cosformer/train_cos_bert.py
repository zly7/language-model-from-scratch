
from datasets import load_dataset, DatasetDict, load_from_disk
from model.model_bert.language_model import  BertConfig,BertLM
# 这句话访问不到是没有访问到module,需要在__init__.py里面引入，可以想象成__init__.py是本题
# from model.model_bert.bert import BERTs
# from model.model_bert.language_model import BERTLM
from transformers import AutoTokenizer
from determine_batch_size import get_batch_size
import torch
def main():
    print("Loading dataset")
    # preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-bert-512-without-test")
    # preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-preprocessed-ws-notext-bert-128-wtest")
    preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-story-bert-512-wtest")
    # preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-story-bert-1024")
    # preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-story-bert-2048")
    # preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-story-bert-4096")
    print("len(preprocessed_splits[validation]"+str(len(preprocessed_splits["validation"])))
    sequence_length = 512
    tokenizer = AutoTokenizer.from_pretrained(f"./tokenizer_save/tokenizer-bert-base-uncased-{sequence_length}")
    print("tokenizer:",str(tokenizer))

    # Create the model
    config = BertConfig(vocab_size=tokenizer.vocab_size, n_embd=768, 
                n_layer=12, n_head=12, dropout=0.1, use_cosformer=True)
    LMmodel = BertLM(config)
    model_size = sum(t.numel() for t in LMmodel.parameters())
    print(f"Bert size: {model_size/1000**2:.1f}M parameters")
    from transformers import DataCollatorForLanguageModeling

    # Create the data collator
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    from TrainArgumentSelf import TrainingArgumentsSelf
    import datetime
    now = datetime.datetime.now()
    date_string = now.strftime("%m-%d-%H-%M")
    batch_size = get_batch_size("base","bert",sequence_length)
    all_batch_size = 512
    all_graient_ac = all_batch_size // batch_size
    device_num = torch.cuda.device_count()
    print("one_node_device_num:",device_num)
    Warning("The all gradient ac is about one node,if you use multi node,please decrease the all_gradient_ac")
    assert device_num >= 1
    gradient_ac = all_graient_ac // device_num
    max_steps = 2e5

    args = TrainingArgumentsSelf(
        output_dir=f"cos_bert_pretrain/wa-{date_string}/",
        # output_dir=f"speed_test/cos_bert/sequence{sequence_length}",
        per_device_train_batch_size=batch_size,   # 16的时候，训练只消耗17.5G显存,24bacth消耗23G,不使用混合精度训练反而24batch还没法用了， 
        per_device_eval_batch_size=int(batch_size*1.5),
        eval_steps=1000 * gradient_ac,
        logging_steps=20 * gradient_ac,
        gradient_accumulation_steps=gradient_ac,
        max_steps=max_steps,   
        num_train_epochs=20,  # 一个epoch差不多是1H，2GPU
        weight_decay=0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-6,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        learning_rate=1e-4,
        save_steps=1_000 * gradient_ac,
        # fp16=False,
        fp16=True,
        report_to="tensorboard",
        train_audit_probability=0,
        test_step=5000*gradient_ac,
        per_device_test_batch_size= batch_size // 2,
        all_test_examples_num=144,
        optimizer_type="adamw",
        sequence_length=sequence_length,
        whether_hg_accelerator=True,
        # resume_from_checkpoint="./compare_bert/cos-05-05-09-54/checkpoint-epoch-accelerate-save-state-27/"
        
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
        test_dataset=preprocessed_splits["test"]
    )
    print("Training model starts")
    trainer.train()


if __name__ == "__main__":
    main()

