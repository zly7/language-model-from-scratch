import torch
from typing import Optional
from TrainArgumentSelf import TrainingArgumentsSelf
from logger import Logger
import random

import json
import os
import time
import math
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
class TrainerSelf():
    def __init__(self, model_name, model, args : TrainingArgumentsSelf, train_dataset, eval_dataset, test_dataset=None,
        tokenizer=None, data_collator=None,
        optimizer: Optional[torch.optim.Optimizer]=None, 
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR]=None,compute_metrics = None):

        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        # self.args.output_dir = os.path.join(self.args.output_dir, self.model_name)

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        # if self.frame == "pytorch": # 这个没意义


        if args.fp16 is True:
            mix_precision = "fp16"
        else:
            mix_precision = "no"
        if self.args.whether_hg_accelerator: 
            from accelerate import Accelerator
            self.accelerator = Accelerator(device_placement=True,gradient_accumulation_steps=args.gradient_accumulation_steps,mixed_precision=mix_precision)
            # self.device =  self.accelerator.cuda.current_device()
            print("self.accelerator.device : " + str(self.accelerator.device))
            self.device = self.accelerator.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        from torch.utils.data.dataloader import DataLoader
        train_dataset.set_format("torch")
        eval_dataset.set_format("torch")
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=self.data_collator)
        
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=self.data_collator)
        if test_dataset is not None:
            test_dataset.set_format("torch") #zly: we don't use collate_fn for test which are not trauncated,这里实际上应该用
            self.test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_test_batch_size, collate_fn=self.data_collator)
            if self.args.test_dataloader_use_accelerate is True and self.args.whether_hg_accelerator and (self.args.test_step is not None):
                self.test_dataloader = self.accelerator.prepare(self.test_dataloader)
        else:
            self.test_dataloader = None
        
        if optimizer is None :
            if self.args.optimizer_type == "adam":
                from torch.optim import Adam
                self.optimizer = Adam(self.get_grouped_params(weight_decay=args.weight_decay), 
                lr=args.learning_rate, weight_decay=args.weight_decay, betas=[args.adam_beta1,args.adam_beta2],eps=args.adam_epsilon)
            # self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=[args.adam_beta1,args.adam_beta2])
            elif self.args.optimizer_type == "sgd":
                from torch.optim import SGD
                self.optimizer = SGD(self.model.parameters(),lr=args.learning_rate,momentum=args.sgd_momentum)
            elif self.args.optimizer_type == "adamw":
                from torch.optim import AdamW
                self.optimizer = AdamW(self.get_grouped_params(weight_decay=args.weight_decay), 
                    lr=args.learning_rate, weight_decay=args.weight_decay, betas=[args.adam_beta1,args.adam_beta2],eps=args.adam_epsilon)
        else:
            self.optimizer = optimizer

        
        if lr_scheduler is None:
            if self.args.lr_scheduler_type == "cosine":
                Warning("No lr_scheduler is provided, using CosineAnnealingLR as default")
                from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
                if args.max_steps >= 1e9:
                    args.max_steps = len(self.train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps
                self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps // args.gradient_accumulation_steps)
                print("lr_schdule_step: " + str(args.max_steps // args.gradient_accumulation_steps) )
            elif self.args.lr_scheduler_type == "constant" or self.args.lr_scheduler_type == "fixed":
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,100,gamma=1)
        else:
            self.lr_scheduler = lr_scheduler
        
        self.logger = Logger(args.report_to,args.output_dir)
        self.model = self.model.to(self.device)
        if self.args.whether_hg_accelerator:
            from accelerate.state import PartialState
            self.state = PartialState()
            if self.state.is_main_process:
                for name, param in self.model.named_parameters():
                    print(name, param.shape)
            self.model, self.optimizer, self.lr_scheduler,self.train_dataloader,self.eval_dataloader = self.accelerator.prepare(model,
                self.optimizer, self.lr_scheduler,self.train_dataloader, self.eval_dataloader)
        else:
            self.state = None
        
        self.best_loss = float("inf")
        self.trainDetailTime = trainDetailTime(self)
        if args.resume_from_checkpoint is not None:
            self.accelerator.load_state(args.resume_from_checkpoint)
        



        
    def train(self):
        self.model.train()
        losses = []
        accuracies = []
        topkaccuracies = []
        completed_steps = 0 
        all_compute_grad_times = 0  # 这个应该和gradient_accumulation_steps解耦
        if self.state is not None:
            print("state process"+ str(self.state.process_index))
        for epoch in range(self.args.num_train_epochs):
            self.time = time.time()
            for step, batch in enumerate(self.train_dataloader):
                self.trainDetailTime.start_train_step()
                if self.args.whether_hg_accelerator is False:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                self.trainDetailTime.start_forward()
                outputs = self.forward_core(batch)
                self.trainDetailTime.end_forward(all_compute_grad_times)
                loss = outputs.loss # loss = outputs.loss loss = output[0] 都可以取到
                losses.append(float(loss)) 
                if hasattr(outputs,'accuracy') and hasattr(outputs,'topkaccuracy'):
                    accuracies.append(float(outputs.accuracy))
                    topkaccuracies.append(float(outputs.topkaccuracy))
                elif self.compute_metrics is not None:
                    accuracies.append(self.compute_metrics(outputs.logits, batch["labels"])["accuracy"])
                    topkaccuracies.append(0.0)
                else:
                    accuracies.append(0.0)
                    topkaccuracies.append(0.0)
                loss = loss / self.args.gradient_accumulation_steps
                self.trainDetailTime.start_backward()
                if self.args.whether_hg_accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                self.trainDetailTime.end_backward(all_compute_grad_times)

                all_compute_grad_times += 1
                if (step+1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.whether_hg_accelerator:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    else:
                        torch.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)
                    
                    self.direct_log_grad(all_compute_grad_times)
                    self.trainDetailTime.start_optimizer()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.trainDetailTime.end_optimizer(all_compute_grad_times)
                    completed_steps += 1
                    if 1-self.args.train_audit_probability < random.random():
                        self.print_debug_info("train",outputs, batch, completed_steps)
                if all_compute_grad_times % self.args.logging_steps == 1:
                    self.average_log_scaler("train","loss",all_compute_grad_times, losses)
                    self.direct_log_scaler("train","lr",all_compute_grad_times, self.lr_scheduler.get_last_lr())
                    if all(x is not None for x  in accuracies):
                        self.direct_log_scaler("train","accuracies",all_compute_grad_times, sum(accuracies)/len(accuracies))
                    if all(x is not None for x  in topkaccuracies):
                        self.direct_log_scaler("train","topkaccuracies",all_compute_grad_times, sum(topkaccuracies)/len(topkaccuracies))
                    losses = []
                    accuracies = []
                    topkaccuracies = []
                
                self.trainDetailTime.end_train_step(all_compute_grad_times,epoch=epoch,step=step)
                if all_compute_grad_times % self.args.save_steps == 1:
                    self.save_model(all_compute_grad_times)
                if all_compute_grad_times % self.args.eval_steps == 20:
                    self.evaluate(all_compute_grad_times)
                if self.args.test_step is not None and all_compute_grad_times % self.args.test_step == 1:
                    self.test(all_compute_grad_times)

                if all_compute_grad_times >= self.args.max_steps:
                    break
            if self.args.whether_hg_accelerator:
                self.accelerator.save_state(output_dir=os.path.join(self.args.output_dir, f"checkpoint-epoch-accelerate-save-state-{epoch}"), state=self.state)
            if all_compute_grad_times >= self.args.max_steps:
                break
                        
        self.save_model(all_compute_grad_times)
    
    def forward_core(self, batch):
        if "huggingface" in self.model_name and "gpt" in self.model_name:
            assert torch.equal(batch["input_ids"], batch["labels"])
            return self.model(batch["input_ids"], labels=batch["labels"])
        elif "self" in self.model_name and "gpt" in self.model_name:
            return self.model(batch["input_ids"], autoShiftLeftAndTrain=True)
        elif "self" in self.model_name and "bert" in self.model_name:
            return self.model(batch["input_ids"], labels =batch["labels"],token_type_ids=batch["token_type_ids"])
        elif "huggingface" in self.model_name and "reformer" in self.model_name:
            return self.model(batch["input_ids"], labels=batch["labels"])
        else :
            Warning("Not Implement")
            return None
        

    def evaluate(self,current_step):
        start_time = time.time()
        self.model.eval()
        losses = []
        accuracies = []
        topkaccuracies = []
        # print("len(self.eval_dataloader)"+str(len(self.eval_dataloader)))
        for step, batch in enumerate(self.eval_dataloader):
            if self.args.whether_hg_accelerator is False:
                batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.forward_core(batch)
            if self.args.whether_hg_accelerator:
                losses.append(self.accelerator.gather(outputs.loss))
                if hasattr(outputs,'accuracy') and outputs.accuracy is not None:
                    accuracies.append(self.accelerator.gather(outputs.accuracy)) # gather 之后获得的是一个tensor组
                elif self.compute_metrics is not None:
                    accuracies.append(self.compute_metrics(self.accelerator.gather(outputs.logits),
                         self.accelerator.gather(batch["labels"]))['accuracy'])
                else:
                    accuracies.append(torch.zeros(size=(1,0)))
                if hasattr(outputs,'topkaccuracy') and outputs.topkaccuracy is not None:
                    topkaccuracies.append(self.accelerator.gather(outputs.topkaccuracy))
                else:
                    topkaccuracies.append(torch.zeros(size=(1,0)))
            else:
                losses.append(outputs.loss.detach().cpu())
                if outputs.accuracy is not None:
                    accuracies.append(outputs.accuracies.detach().cpu())
                if outputs.topkaccuracy is not None:
                    topkaccuracies.append(outputs.topkaccuracy)
        
        del batch
        del outputs
        
        self.average_log_scaler("evaluate","loss",current_step, torch.stack(losses).flatten()) 
        if all(x is not None for x  in accuracies):
            self.average_log_scaler("evaluate","accuracies",current_step,torch.stack(accuracies).flatten())
        if all(x is not None for x  in topkaccuracies):
            self.average_log_scaler("evaluate","topkaccuracies",current_step, torch.stack(topkaccuracies).flatten())
        evaluate_time = time.time() - start_time
        self.direct_log_scaler("evaluate",f"inference_time",current_step, evaluate_time)
        self.direct_log_scaler("evaluate",f"inference_speed(s-per-step-per-gpu))-bs-{self.args.per_device_eval_batch_size}",current_step, evaluate_time/len(self.eval_dataloader))
        self.direct_log_scaler("evaluate",f"inference_per_step_spend_time(step-per-s-per-gpu)-bs-{self.args.per_device_eval_batch_size}",current_step, len(self.eval_dataloader)/evaluate_time)
        self.direct_log_scaler("evaluate",f"inference_example_per_second(example-per-s-per-gpu)",current_step, len(self.eval_dataloader) * self.args.per_device_eval_batch_size / evaluate_time) # 这几个计算和train还有一些不同，涉及到GPU的个数
        self.model.train()
        
    
    def test(self,current_step):
        if self.tokenizer is None:
            Warning("Can't do test without tokenizer")
            return
        self.model.eval()
        answer = []
        uw_model = self.accelerator.unwrap_model(self.model)
        if "visualize" in self.model_name and "gpt" in self.model_name:
            visualize_dir = os.path.join(self.args.output_dir,"visualize")
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
                        plt.savefig(os.path.join(visualize_dir,f"batch_{step}_index_{i}_heads_{j}_attention.png"))
                        plt.cla()
                    
                    for j in range(len(heads_num)):
                        plt.figure(figsize=(50, 50))
                        sns.heatmap(att[i][j].detach().numpy(), cmap='YlGnBu', annot=True)
                        plt.xticks(np.arange(0, sequence_l, 25))
                        plt.yticks(np.arange(0, sequence_l, 25))
                        # 保存图像
                        plt.savefig(os.path.join(visualize_dir,f"batch_{step}_index_{i}_heads_{j}_attention_before_softmax.png"))
                        plt.cla()



                
        if "gpt" in self.model_name and "visualize" not in self.model_name:
            for step, batch in enumerate(self.test_dataloader):
                if step*self.args.per_device_test_batch_size > self.args.all_test_examples_num:
                    break
                batch["input_ids"] = batch["input_ids"].to(self.accelerator.device)
                prediction_label_length = batch["prediction_labels"].shape[1]
                with torch.no_grad():
                    if "gpt" in self.model_name or "causal" in self.model_name:
                        outputs_ids = uw_model.generate(batch["input_ids"], max_new_tokens=prediction_label_length, top_k = 7)
                    else:
                        Warning("Not Implement")
                b,t=batch["input_ids"].shape
                for i in range(b):
                    one_answer_dic = {}
                    one_answer_dic["propmt text"] = self.tokenizer.decode(batch["input_ids"][i])
                    one_answer_dic["generated text"] = self.tokenizer.decode(outputs_ids[i])
                    one_answer_dic["origin answer"] = self.tokenizer.decode(batch["prediction_labels"][i])
                    answer.append(one_answer_dic)
        elif "bert" in self.model_name:
            how_many_masks  = 0
            how_many_right = 0
            for step, batch in enumerate(self.test_dataloader):
                if step*self.args.per_device_test_batch_size > self.args.all_test_examples_num:
                    break
                if not self.args.whether_hg_accelerator:
                    batch["input_ids"] = batch["input_ids"].to(self.device)
                    batch["token_type_ids"] = batch["token_type_ids"].to(self.device)
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
                            if batch["labels"][i][label_index][current_index] == temp_generated_text_ids[current_index]:
                                how_many_right += 1
                            how_many_masks += 1
                            current_index += 1
                            j += 4
                        j += 1
                    one_answer_dic["compared answer"] = self.tokenizer.decode(temp_input_ids)
                    answer.append(one_answer_dic)
            self.direct_log_scaler("test", "accuracies", current_step, how_many_right/how_many_masks)
        else:
            Warning("Not Implement")

        with open(os.path.join(self.args.output_dir,f"test-gernerate-answer-{str(current_step)}-process-{self.state.process_index}.json"),"w") as f:
            json.dump(answer,f)
        self.model.train()
        return None


    def average_log_scaler(self, stage, name, step, scalers, whether_print = True):  # 这里输入最好是tensor flatten的
        if (self.args.whether_hg_accelerator and self.accelerator.is_main_process) or (not self.args.whether_hg_accelerator):
            # print(scalers)
            if isinstance(scalers, (list, tuple)): # list里面是一个二维tensor会导致无法转换
                scalers = torch.tensor(scalers, dtype=torch.float32)
            to_log = torch.mean(scalers)

            if whether_print:
                print(f"Stage {stage}, Step {step}: {name}={to_log.item()}")
            self.logger.log_scaler(f"{name}/{stage}", float(to_log.item()), step)
    
    def direct_log_scaler(self, stage, name, step, scaler, whether_print = True):  # 这里的输入最好是一个元素的tensor
        if (self.args.whether_hg_accelerator and self.accelerator.is_main_process) or (not self.args.whether_hg_accelerator):
            if isinstance(scaler, (list, tuple)):
                scaler = scaler[0]
            if isinstance(scaler, (torch.Tensor,np.ndarray)):  # 这里有一种可能是，scaler是一个tensor，但是这个tensor只有一个元素
                if scaler.numel() == 1:
                    scaler = scaler.item()
                else:
                    Warning("The scaler in direct log fuction is not a scaler")
                    scaler = scaler[0].item()
            
            if whether_print:
                print(f"Stage {stage}, Step {step}: {name}={scaler}")
            self.logger.log_scaler(f"{name}/{stage}", float(scaler), step)
    
    def save_model(self, step):
        if (self.args.whether_hg_accelerator and self.accelerator.is_main_process) or (not self.args.whether_hg_accelerator):
            if self.args.whether_hg_accelerator:
                model_unwrapped = self.accelerator.unwrap_model(self.model)
            else:
                model_unwrapped = self.model
            model_unwrapped.save_pretrained(f"{self.args.output_dir}/checkpoint-{step}")
            # self.accelerator.save(self.model, f"{self.args.output_dir}/checkpoint-{step}/model.pkl")
            with open(f"{self.args.output_dir}/checkpoint-{step}/training_args.json","w") as f:
                json.dump(self.args.to_dict(), f)
            print(f"Saving model checkpoint to {self.args.output_dir}/checkpoint-{step}")
    
    def direct_log_grad(self,step):
        for name, param in self.model.named_parameters():
            self.direct_log_scaler(stage="grad",name = name, step=step, scaler=torch.norm(param.grad,p="fro"),whether_print=False)
            # if "wpe" in name:
            #     torch.set_printoptions(edgeitems=1000, precision=5, linewidth=100)
            #     print("wpe grad:"+str(param.grad))
    
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
    
    def get_grouped_params(self, no_decay=["bias", "LayerNorm.weight","ln","norm"],weight_decay=0.01):
        from product_key_memory import fetch_pkm_value_parameters
        params_with_wd, params_without_wd = [], []
        pkm_parameters, other_parameters = fetch_pkm_value_parameters(self.model)
        for n, p in self.model.named_parameters():
            if any(nd in n for nd in no_decay):
                params_without_wd.append(p)
            else:
                params_with_wd.append(p)
        return [
            {"params": params_with_wd, "weight_decay": weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
            {"params": pkm_parameters, "lr": 1e-2},
        ]


class trainDetailTime:
    def __init__(self,trainer:TrainerSelf) -> None:
        self.start_forward_time = -1
        self.end_forward_time = -1
        self.start_backward_time = -1
        self.end_backward_time = -1
        self.start_optimizer_time = -1
        self.end_optimizer_time = -1
        self.start_train_step_time = -1
        self.end_train_step_time = -1
        self.trainer = trainer
        self.training_step_time_list = []
    
    def start_forward(self):
        self.start_forward_time = time.time()
    
    def end_forward(self,step):
        assert self.start_forward_time != -1
        self.end_forward_time = time.time()
        self.trainer.direct_log_scaler(stage="train",name="forward-time",step=step,scaler=self.end_forward_time-self.start_forward_time,whether_print=False)
        self.start_forward_time = -1
        self.end_backward_time = -1
    
    def start_backward(self):
        self.start_backward_time = time.time()
    
    def end_backward(self,step):
        assert self.start_backward_time != -1
        self.end_backward_time = time.time()
        self.trainer.direct_log_scaler(stage="train",name="backward-time",step=step,scaler=self.end_backward_time-self.start_backward_time,whether_print=False)
        self.start_backward_time = -1
        self.end_backward_time = -1
    
    def start_optimizer(self):
        self.start_optimizer_time = time.time()
    
    def end_optimizer(self,step):
        assert self.start_optimizer_time != -1
        self.end_optimizer_time = time.time()
        self.trainer.direct_log_scaler(stage="train",name="optimizer-time",step=step,scaler=self.end_optimizer_time-self.start_optimizer_time, whether_print=False)
        self.start_optimizer_time = -1
        self.end_optimizer_time = -1
    
    def start_train_step(self):
        self.start_train_step_time = time.time()
    
    def end_train_step(self,all_compute_grad_times,epoch,step):
        assert self.start_train_step_time != -1
        self.end_train_step_time = time.time()
        self.training_step_time_list.append(self.end_train_step_time-self.start_train_step_time)
        if len(self.training_step_time_list) == self.trainer.args.gradient_accumulation_steps:
            train_average_time =  torch.mean(torch.tensor(self.training_step_time_list))  # 一个梯度累积的时间
            self.trainer.direct_log_scaler("train",f"accumulation_step_spends_second(s-per-step)-bs-{self.trainer.args.per_device_train_batch_size}",all_compute_grad_times, train_average_time, whether_print=False)
            step_per_second = self.trainer.args.gradient_accumulation_steps / train_average_time  # todo-这里要消除GPU个数的影响
            self.trainer.direct_log_scaler("train",f"accumulation_step_per_second(step-per-s)-bs-{self.trainer.args.per_device_train_batch_size}",all_compute_grad_times, step_per_second, whether_print=False)
            example_per_second =  self.trainer.args.per_device_train_batch_size * self.trainer.args.gradient_accumulation_steps / train_average_time
            self.trainer.direct_log_scaler("train","train_example_per_second(example-per-s)",all_compute_grad_times, example_per_second) # 最核心关注速度指标，训练一个seque
            self.trainer.direct_log_scaler("train","epoch",all_compute_grad_times, epoch+step/len(self.trainer.train_dataloader))   
            self.training_step_time_list = []



    