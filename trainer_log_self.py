# 这个是废弃的log版本，现在用的是accelerate的log,后来还是选择自己log,更加灵活
import torch
from typing import Optional
from TrainArgumentSelf import TrainingArgumentsSelf
from logger import Logger
import random

import json
import os
class TrainerSelf():
    def __init__(self, model_name, model, args : TrainingArgumentsSelf, train_dataset, eval_dataset, test_dataset=None,
        tokenizer=None, data_collator=None,
        optimizer: Optional[torch.optim.Optimizer]=None, 
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR]=None):

        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        # if self.frame == "pytorch": # 这个没意义
        # if args.whether_hg_accelerator: # 暂时完全用accelerator
        from accelerate import Accelerator
        self.accelerator = Accelerator(device_placement=True,gradient_accumulation_steps=args.gradient_accumulation_steps)
        # self.device =  self.accelerator.cuda.current_device()
        print("self.accelerator.device : " + str(self.accelerator.device))
        self.device = self.accelerator.device
        # else:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     self.model.to(self.device)
        
        from torch.utils.data.dataloader import DataLoader
        train_dataset.set_format("torch")
        eval_dataset.set_format("torch")
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=self.data_collator)
        
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=self.data_collator)
        if test_dataset is not None:
            test_dataset.set_format("torch") #zly: wo don't use collate_fn for test which are not trauncated
            self.test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_test_batch_size)
        else:
            self.test_dataloader = None
        
        if optimizer is None :
            Warning("No optimizer is provided, using AdamW as default")
            from transformers import AdamW
            self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            self.optimizer = optimizer

        
        if lr_scheduler is None:
            Warning("No lr_scheduler is provided, using CosineAnnealingLR as default")
            from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
            if args.max_steps <= 0:
                args.max_steps = len(self.train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
        else:
            self.lr_scheduler = lr_scheduler
        
        self.logger = Logger(args.report_to,args.output_dir)
        self.model = self.model.to(self.device)
        self.model, self.optimizer, self.lr_scheduler,self.train_dataloader,self.eval_dataloader = self.accelerator.prepare(model,
            self.optimizer, self.lr_scheduler,self.train_dataloader, self.eval_dataloader)
        # if self.test_dataloader is not None:
        #     self.test_dataloader = self.accelerator.prepare(self.test_dataloader) # 每个进程的test是一样的
        from accelerate.state import PartialState
        self.state = PartialState()
        
        self.best_loss = float("inf")


        
    def train(self):
        self.model.train()
        losses = []
        completed_steps = 0 
        all_compute_grad_times = 0  # 这个应该和gradient_accumulation_steps解耦

        print("state process"+ str(self.state.process_index))
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.forward_core(batch)
                    loss = outputs[0] # loss = outputs.loss loss = output[0]
                    losses.append(loss)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    all_compute_grad_times += 1
                    if step % self.args.gradient_accumulation_steps == 0:
                        completed_steps += 1
                        if 1-self.args.train_audit_probability < random.random():
                            self.print_debug_info("train",outputs, batch, completed_steps)
                    if all_compute_grad_times % self.args.logging_steps == 1:
                        self.log("train","loss",all_compute_grad_times, losses)
                        self.log("train","lr",all_compute_grad_times, [self.lr_scheduler.get_last_lr()])
                        losses = []
                    if all_compute_grad_times % self.args.save_steps == 1:
                        self.save_model(all_compute_grad_times)
                    if all_compute_grad_times % self.args.eval_steps == 1:
                        self.evaluate(all_compute_grad_times)
                    if hasattr(self.args,"test_step") and all_compute_grad_times % self.args.test_step == 1:
                        self.test(all_compute_grad_times)
                        
            self.save_model(all_compute_grad_times)
    
    def forward_core(self, batch):
        if "huggingface" in self.model_name and "gpt" in self.model_name:
            return self.model(batch["input_ids"], labels=batch["input_ids"])
        elif "self" in self.model_name and "gpt" in self.model_name:
            return self.model(batch["input_ids"], autoShiftLeftAndTrain=True)
        else :
            Warning("Not Implement")
            return None
        

    def evaluate(self,current_step):
        self.model.eval()
        losses = []
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.forward_core(batch)
            losses.append(self.accelerator.gather(outputs.loss))
        self.log("evaluate","loss",current_step, losses)
        self.model.train()
        
    
    def test(self,current_step):
        if self.tokenizer is None:
            Warning("Can't do test without tokenizer")
            return
        self.model.eval()
        answer = []
        uw_model = self.accelerator.unwrap_model(self.model)
        for step, batch in enumerate(self.test_dataloader):
            if step*self.args.per_device_test_batch_size > self.args.all_test_examples_num:
                break
            batch["input_ids"] = batch["input_ids"].to(self.accelerator.device)
            with torch.no_grad():
                if "gpt" in self.model_name or "causal" in self.model_name:
                    outputs_ids = uw_model.generate(batch["input_ids"], max_new_tokens=128, top_k = 7)
                else:
                    Warning("Not Implement")
            b,t=batch["input_ids"].shape
            for i in range(b):
                one_answer_dic = {}
                one_answer_dic["propmt text"] = self.tokenizer.decode(batch["input_ids"][i])
                one_answer_dic["generated text"] = self.tokenizer.decode(outputs_ids[i])
                one_answer_dic["origin answer"] = self.tokenizer.decode(batch["overflowing_tokens"][i])
                answer.append(one_answer_dic)
            # print("finish one batch of test generation") # 1s 2个

        with open(os.path.join(self.args.output_dir,f"test-gernerate-answer-{str(current_step)}-process-{self.state.process_index}.json"),"w") as f:
            json.dump(answer,f)
        self.model.train()
        return None


    def log(self, stage, name, step, scalers):
        if self.accelerator.is_main_process:
            try:
                loss = torch.mean(torch.cat(scalers))
            except:
                loss = torch.mean(torch.stack(scalers))
            # try:
            #     perplexity = torch.exp(loss)
            # except OverflowError:
            #     perplexity = float("inf")
            print(f"Stage {stage}, Step {step}: loss={loss.item()}")
            self.logger.log_scaler(f"{name}/{stage}", loss, step)
            # self.logger.log_scaler(f"{name}-Perplexity/{stage}", perplexity, step)
    
    def save_model(self, step):
        if self.accelerator.is_main_process:
            model_unwrapped = self.accelerator.unwrap_model(self.model)
            model_unwrapped.save_pretrained(f"{self.args.output_dir}/checkpoint-{step}")
            # self.accelerator.save(self.model, f"{self.args.output_dir}/checkpoint-{step}/model.pkl")
            torch.save(self.args, f"{self.args.output_dir}/checkpoint-{step}/training_args.bin")
            print(f"Saving model checkpoint to {self.args.output_dir}/checkpoint-{step}")
    
    def print_debug_info(self,stage:str,outputs,batch,completed_steps):
        if self.accelerator.is_main_process:
            b,t = batch["input_ids"].shape
            # get the probability of the right token
            targets =  (batch["input_ids"][:, 1:]).contiguous()
            probs = torch.softmax(outputs.logits, dim=-1)
            probs = probs.view(b*t, -1)
            targets = targets.view(b*(t-1))
            targets_probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))
            targets_probs = targets_probs.reshape(b, t-1) 

            answer = []
            for i in range(b):
                one_answer_dic = {}
                one_answer_dic["propmt text"] = self.tokenizer.decode(batch["input_ids"][i])
                one_answer_dic["propmt text corresponding probability"] =  [round(prob, 5) for prob in targets_probs[i].tolist()]
                one_answer_dic["generated max probability text"] = self.tokenizer.decode(outputs.logits[i].argmax(dim=-1))
                answer.append(one_answer_dic)
            with open(os.path.join(self.args.output_dir,f"{stage}-debug-info-{str(completed_steps)}.json"),"w") as f:
                json.dump(answer,f)


            

    

    def inspect_GPU_situation(self):  # 这个代码有点问题，就是不会显示另外一个GPU的使用
        if self.accelerator.is_local_main_process:
            num_gpus = torch.cuda.device_count()
            print("Number of available GPUs:", num_gpus)
            for i in range(num_gpus):
                device = torch.device("cuda:" + str(i))
                print("GPU device name:", torch.cuda.get_device_name(device),end=f'{i}')
                print("GPU memory usage:")
                print(torch.cuda.memory_allocated(device)/1024**2, "MB allocated")
                print(torch.cuda.memory_cached(device)/1024**2, "MB cached")





    