import torch
from typing import Optional
from TrainArgumentSelf import TrainingArgumentsSelf
from logger import Logger
import random

import json
import os
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
class TrainerSelf():
    def __init__(self, model_name, model, args : TrainingArgumentsSelf, train_dataset, eval_dataset, test_dataset=None,
        tokenizer=None, data_collator=None,
        optimizer: Optional[torch.optim.Optimizer]=None, 
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR]=None):

        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        # self.args.output_dir = os.path.join(self.args.output_dir, self.model_name)

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        # if self.frame == "pytorch": # 这个没意义
        # if args.whether_hg_accelerator: # 暂时完全用accelerator
        if args.fp16 is True:
            mix_precision = "fp16"
        else:
            mix_precision = "no"
        from accelerate import Accelerator
        self.accelerator = Accelerator(device_placement=True,gradient_accumulation_steps=args.gradient_accumulation_steps,mixed_precision=mix_precision)
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
            test_dataset.set_format("torch") #zly: we don't use collate_fn for test which are not trauncated,这里实际上应该用
            self.test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_test_batch_size, collate_fn=self.data_collator)
            if self.args.test_dataloader_use_accelerate is True:
                self.test_dataloader = self.accelerator.prepare(self.test_dataloader)
        else:
            self.test_dataloader = None
        
        if optimizer is None :
            Warning("No optimizer is provided, using Adam as default")
            from torch.optim import Adam
            self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=[args.adam_beta1,args.adam_beta2],eps=args.adam_epsilon)
            # self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=[args.adam_beta1,args.adam_beta2])
        else:
            self.optimizer = optimizer

        
        if lr_scheduler is None:
            Warning("No lr_scheduler is provided, using CosineAnnealingLR as default")
            from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
            if args.max_steps >= 1e9:
                args.max_steps = len(self.train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
        else:
            self.lr_scheduler = lr_scheduler
        
        self.logger = Logger(args.report_to,args.output_dir)
        self.model = self.model.to(self.device)
        self.model, self.optimizer, self.lr_scheduler,self.train_dataloader,self.eval_dataloader = self.accelerator.prepare(model,
            self.optimizer, self.lr_scheduler,self.train_dataloader, self.eval_dataloader)
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
            self.time = time.time()
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
                        self.average_log_scaler("train","loss",all_compute_grad_times, losses)
                        self.direct_log_scaler("train","lr",all_compute_grad_times, self.lr_scheduler.get_last_lr())
                        train_average_time = (time.time() - self.time) / self.args.logging_steps  # 一个梯度累积的时间
                        self.direct_log_scaler("train",f"accumulation_step_spends_second(s/step)-bs-{self.args.per_device_train_batch_size}",all_compute_grad_times, train_average_time)
                        step_per_second = self.args.gradient_accumulation_steps / train_average_time
                        self.direct_log_scaler("train",f"faccumulation_step_per_seconde(step/s)-bs-{self.args.per_device_train_batch_size}",all_compute_grad_times, step_per_second)
                        example_per_second =  self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps / train_average_time
                        self.direct_log_scaler("train","train_example_per_second(example/s)",all_compute_grad_times, example_per_second) # 最核心关注速度指标，训练一个seque
                        self.time = time.time()
                        losses = []
                    if all_compute_grad_times % self.args.save_steps == 1:
                        self.save_model(all_compute_grad_times)
                    if all_compute_grad_times % self.args.eval_steps == 1:
                        self.evaluate(all_compute_grad_times)
                    if self.args.test_step is not None and all_compute_grad_times % self.args.test_step == 1:
                        self.test(all_compute_grad_times)

                    if all_compute_grad_times >= self.args.max_steps:
                        break
            self.accelerator.save_state(output_dir=os.join(self.args.output_dir, f"checkpoint-epoch-accelerate-save-state-{epoch}"), state=self.state)
            if all_compute_grad_times >= self.args.max_steps:
                break
                        
        self.save_model(all_compute_grad_times)
    
    def forward_core(self, batch):
        if "huggingface" in self.model_name and "gpt" in self.model_name:
            return self.model(batch["input_ids"], labels=batch["input_ids"])
        elif "self" in self.model_name and "gpt" in self.model_name:
            return self.model(batch["input_ids"], autoShiftLeftAndTrain=True)
        elif "self" in self.model_name and "bert" in self.model_name:
            return self.model(batch["input_ids"], labels =batch["labels"],token_type_ids=batch["token_type_ids"])
        else :
            Warning("Not Implement")
            return None
        

    def evaluate(self,current_step):
        start_time = time.time()
        self.model.eval()
        losses = []
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.forward_core(batch)
            losses.append(self.accelerator.gather(outputs.loss))
        self.average_log_scaler("evaluate","loss",current_step, losses)
        evaluate_time = time.time() - start_time
        self.direct_log_scaler("evaluate",f"inference_speed(s/step)-bs-{self.args.per_device_eval_batch_size}",current_step, evaluate_time/len(self.eval_dataloader))
        self.direct_log_scaler("evaluate",f"inference_per_step_spend_time(step/s)-bs-{self.args.per_device_eval_batch_size}",current_step, len(self.eval_dataloader)/evaluate_time)
        self.direct_log_scaler("evaluate","evaluate_example_per_second(example/s)",current_step, len(self.eval_dataloader) * self.args.per_device_eval_batch_size / evaluate_time)
        self.model.train()
        
    
    def test(self,current_step):
        if self.tokenizer is None:
            Warning("Can't do test without tokenizer")
            return
        self.model.eval()
        answer = []
        uw_model = self.accelerator.unwrap_model(self.model)
        if "visualize" in self.model_name and "gpt" in self.model_name:
            visualize_dir = os.join(self.args.output_dir,"visualize")
            assert uw_model.config.visualize == True
            for step, batch in enumerate(self.test_dataloader):
                outputs = uw_model(batch["input_ids"], labels=batch["input_ids"])
                layers_num = len(outputs.attentions)
                batch,heads_num,sequence_l,_sequence_l = outputs.attentions[0].shape
                _batch,_heads_num,_sequence_l,embadding_size = outputs.keys[0].shape
                
                assert batch == _batch
                assert heads_num == _heads_num
                for i in range(len(batch)):
                    q = outputs.queries[i]
                    k = outputs.keys[i]
                    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(embadding_size))
                    for j in range(len(heads_num)):
                        plt.figure(figsize=(50, 50))
                        sns.heatmap(outputs.attentions[i][j].detach().numpy(), cmap='YlGnBu', annot=True)
                        plt.xticks(np.arange(0, sequence_l, 25))
                        plt.yticks(np.arange(0, sequence_l, 25))
                        # 保存图像
                        plt.savefig(os.join(visualize_dir,f"batch_{step}_index_{i}_heads_{j}_attention.png"))
                        plt.cla()
                    
                    for j in range(len(heads_num)):
                        plt.figure(figsize=(50, 50))
                        sns.heatmap(att[i][j].detach().numpy(), cmap='YlGnBu', annot=True)
                        plt.xticks(np.arange(0, sequence_l, 25))
                        plt.yticks(np.arange(0, sequence_l, 25))
                        # 保存图像
                        plt.savefig(os.join(visualize_dir,f"batch_{step}_index_{i}_heads_{j}_attention_before_softmax.png"))
                        plt.cla()



                
        if "gpt" in self.model_name:
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
        elif "bert" in self.model_name:
            for step, batch in enumerate(self.test_dataloader):
                if step*self.args.per_device_test_batch_size > self.args.all_test_examples_num:
                    break
                batch["input_ids"] = batch["input_ids"].to(self.accelerator.device)
                batch["token_type_ids"] = batch["token_type_ids"].to(self.accelerator.device)
                outputs = self.model(batch["input_ids"], token_type_ids=batch["token_type_ids"],labels = None)
                b,t=batch["input_ids"].shape 
                # logits shape [batch, sequence length, vocab_size]
                left_bracket_index = self.tokenizer.convert_tokens_to_ids("[")
                right_bracket_index = self.tokenizer.convert_tokens_to_ids("]")
                comma_index = self.tokenizer.convert_tokens_to_ids(",")
                for i in range(b):
                    one_answer_dic = {}
                    one_answer_dic["propmt text"] = self.tokenizer.decode(batch["input_ids"][i])
                    temp_input_ids = batch["input_ids"][i].tolist()
                    label_index = torch.where(batch["input_ids"][i] == self.tokenizer.mask_token_id)[0]
                    temp_generated_text_ids = torch.argmax(outputs["logits"][i][label_index],dim=-1)
                    one_answer_dic["generated text"] = self.tokenizer.decode(temp_generated_text_ids)
                    one_answer_dic["origin answer"] = self.tokenizer.decode(batch["labels"][i][label_index])
                    current_index = 0
                    j = 0
                    while j < len(temp_input_ids):
                        if temp_input_ids[j] == self.tokenizer.mask_token_id:
                            temp_input_ids.pop(j)
                            temp_input_ids[j:j] = [left_bracket_index,batch["labels"][i][label_index][current_index],comma_index,temp_generated_text_ids[current_index],right_bracket_index]
                            current_index += 1
                            j += 4
                        j += 1
                    one_answer_dic["compared answer"] = self.tokenizer.decode(temp_input_ids)
                    answer.append(one_answer_dic)
        else:
            Warning("Not Implement")

        with open(os.path.join(self.args.output_dir,f"test-gernerate-answer-{str(current_step)}-process-{self.state.process_index}.json"),"w") as f:
            json.dump(answer,f)
        self.model.train()
        return None


    def average_log_scaler(self, stage, name, step, scalers):
        if self.accelerator.is_main_process:
            try:
                to_log = torch.mean(torch.cat(scalers))
            except:
                to_log = torch.mean(torch.stack(scalers))
            # try:
            #     perplexity = torch.exp(loss)
            # except OverflowError:
            #     perplexity = float("inf")
            print(f"Stage {stage}, Step {step}: {name}={to_log.item()}")
            self.logger.log_scaler(f"{name}/{stage}", to_log, step)
            # self.logger.log_scaler(f"{name}-Perplexity/{stage}", perplexity, step)
    
    def direct_log_scaler(self, stage, name, step, scaler):
        if self.accelerator.is_main_process:
            try:
                if len(scaler) == 1:
                    scaler = scaler[0]
                elif len(scaler) > 1:
                    Warning("the scaler should be one number")
            except:
                pass
            
            print(f"Stage {stage}, Step {step}: {name}={scaler}")
            self.logger.log_scaler(f"{name}/{stage}", scaler, step)
    
    def save_model(self, step):
        if self.accelerator.is_main_process:
            model_unwrapped = self.accelerator.unwrap_model(self.model)
            model_unwrapped.save_pretrained(f"{self.args.output_dir}/checkpoint-{step}")
            # self.accelerator.save(self.model, f"{self.args.output_dir}/checkpoint-{step}/model.pkl")
            with open(f"{self.args.output_dir}/checkpoint-{step}/training_args.json","w") as f:
                json.dump(self.args.to_dict(), f)
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





    