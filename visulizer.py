# import torch
# from typing import Optional
# from TrainArgumentSelf import TrainingArgumentsSelf
# from logger import Logger
# import random

# import json
# import os
# import time
# class TrainerSelf():
#     def __init__(self, model_name, model, args : TrainingArgumentsSelf, train_dataset, eval_dataset, test_dataset=None,
#         tokenizer=None, data_collator=None,
#         optimizer: Optional[torch.optim.Optimizer]=None, 
#         lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR]=None):

#         self.model_name = model_name
#         self.model = model
#         self.tokenizer = tokenizer
#         self.args = args
#         # self.args.output_dir = os.path.join(self.args.output_dir, self.model_name)

#         self.data_collator = data_collator
#         self.train_dataset = train_dataset
#         self.eval_dataset = eval_dataset
#         # if self.frame == "pytorch": # 这个没意义
#         # if args.whether_hg_accelerator: # 暂时完全用accelerator
#         if args.fp16 is True:
#             mix_precision = "fp16"
#         else:
#             mix_precision = "no"
#         from accelerate import Accelerator
#         self.accelerator = Accelerator(device_placement=True,gradient_accumulation_steps=args.gradient_accumulation_steps,mixed_precision=mix_precision)
#         # self.device =  self.accelerator.cuda.current_device()
#         print("self.accelerator.device : " + str(self.accelerator.device))
#         self.device = self.accelerator.device
#         # else:
#         #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         #     self.model.to(self.device)
        
#         from torch.utils.data.dataloader import DataLoader
#         train_dataset.set_format("torch")
#         eval_dataset.set_format("torch")
#         self.train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=self.data_collator)
        
#         self.eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=self.data_collator)
#         if test_dataset is not None:
#             test_dataset.set_format("torch") #zly: we don't use collate_fn for test which are not trauncated,这里实际上应该用
#             self.test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_test_batch_size, collate_fn=self.data_collator)
#             if self.args.test_dataloader_use_accelerate is True:
#                 self.test_dataloader = self.accelerator.prepare(self.test_dataloader)
#         else:
#             self.test_dataloader = None
        
#         if optimizer is None :
#             Warning("No optimizer is provided, using Adam as default")
#             from torch.optim import Adam
#             self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=[args.adam_beta1,args.adam_beta2],eps=args.adam_epsilon)
#             # self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=[args.adam_beta1,args.adam_beta2])
#         else:
#             self.optimizer = optimizer

        
#         if lr_scheduler is None:
#             Warning("No lr_scheduler is provided, using CosineAnnealingLR as default")
#             from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
#             if args.max_steps >= 1e9:
#                 args.max_steps = len(self.train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps
#             self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
#         else:
#             self.lr_scheduler = lr_scheduler
        
#         self.logger = Logger(args.report_to,args.output_dir)
#         self.model = self.model.to(self.device)
#         self.model, self.optimizer, self.lr_scheduler,self.train_dataloader,self.eval_dataloader = self.accelerator.prepare(model,
#             self.optimizer, self.lr_scheduler,self.train_dataloader, self.eval_dataloader)
#         from accelerate.state import PartialState
#         self.state = PartialState()
        
#         self.best_loss = float("inf")


        
    
#     def test(self,current_step):
#         if self.tokenizer is None:
#             Warning("Can't do test without tokenizer")
#             return
#         self.model.eval()
#         answer = []
#         uw_model = self.accelerator.unwrap_model(self.model)
#         if "gpt" in self.model_name:
#             for step, batch in enumerate(self.test_dataloader):
#                 if step*self.args.per_device_test_batch_size > self.args.all_test_examples_num:
#                     break
#                 batch["input_ids"] = batch["input_ids"].to(self.accelerator.device)
#                 with torch.no_grad():
#                     if "gpt" in self.model_name or "causal" in self.model_name:
#                         outputs_ids = uw_model.generate(batch["input_ids"], max_new_tokens=128, top_k = 7)
#                     else:
#                         Warning("Not Implement")
#                 b,t=batch["input_ids"].shape
#                 for i in range(b):
#                     one_answer_dic = {}
#                     one_answer_dic["propmt text"] = self.tokenizer.decode(batch["input_ids"][i])
#                     one_answer_dic["generated text"] = self.tokenizer.decode(outputs_ids[i])
#                     one_answer_dic["origin answer"] = self.tokenizer.decode(batch["overflowing_tokens"][i])
#                     answer.append(one_answer_dic)
#         elif "bert" in self.model_name:
#             for step, batch in enumerate(self.test_dataloader):
#                 if step*self.args.per_device_test_batch_size > self.args.all_test_examples_num:
#                     break
#                 batch["input_ids"] = batch["input_ids"].to(self.accelerator.device)
#                 batch["token_type_ids"] = batch["token_type_ids"].to(self.accelerator.device)
#                 outputs = self.model(batch["input_ids"], token_type_ids=batch["token_type_ids"],labels = None)
#                 b,t=batch["input_ids"].shape 
#                 # logits shape [batch, sequence length, vocab_size]
#                 left_bracket_index = self.tokenizer.convert_tokens_to_ids("[")
#                 right_bracket_index = self.tokenizer.convert_tokens_to_ids("]")
#                 comma_index = self.tokenizer.convert_tokens_to_ids(",")
#                 for i in range(b):
#                     one_answer_dic = {}
#                     one_answer_dic["propmt text"] = self.tokenizer.decode(batch["input_ids"][i])
#                     temp_input_ids = batch["input_ids"][i].tolist()
#                     label_index = torch.where(batch["input_ids"][i] == self.tokenizer.mask_token_id)[0]
#                     temp_generated_text_ids = torch.argmax(outputs["logits"][i][label_index],dim=-1)
#                     one_answer_dic["generated text"] = self.tokenizer.decode(temp_generated_text_ids)
#                     one_answer_dic["origin answer"] = self.tokenizer.decode(batch["labels"][i][label_index])
#                     current_index = 0
#                     j = 0
#                     while j < len(temp_input_ids):
#                         if temp_input_ids[j] == self.tokenizer.mask_token_id:
#                             temp_input_ids.pop(j)
#                             temp_input_ids[j:j] = [left_bracket_index,batch["labels"][i][label_index][current_index],comma_index,temp_generated_text_ids[current_index],right_bracket_index]
#                             current_index += 1
#                             j += 4
#                         j += 1
#                     one_answer_dic["compared answer"] = self.tokenizer.decode(temp_input_ids)
#                     answer.append(one_answer_dic)
#         else:
#             Warning("Not Implement")

#         with open(os.path.join(self.args.output_dir,f"test-gernerate-answer-{str(current_step)}-process-{self.state.process_index}.json"),"w") as f:
#             json.dump(answer,f)
#         self.model.train()
#         return None


#     def average_log_scaler(self, stage, name, step, scalers):
#         if self.accelerator.is_main_process:
#             try:
#                 to_log = torch.mean(torch.cat(scalers))
#             except:
#                 to_log = torch.mean(torch.stack(scalers))
#             # try:
#             #     perplexity = torch.exp(loss)
#             # except OverflowError:
#             #     perplexity = float("inf")
#             print(f"Stage {stage}, Step {step}: {name}={to_log.item()}")
#             self.logger.log_scaler(f"{name}/{stage}", to_log, step)
#             # self.logger.log_scaler(f"{name}-Perplexity/{stage}", perplexity, step)
    
#     def direct_log_scaler(self, stage, name, step, scaler):
#         if self.accelerator.is_main_process:
#             try:
#                 if len(scaler) == 1:
#                     scaler = scaler[0]
#                 elif len(scaler) > 1:
#                     Warning("the scaler should be one number")
#             except:
#                 pass
            
#             print(f"Stage {stage}, Step {step}: {name}={scaler}")
#             self.logger.log_scaler(f"{name}/{stage}", scaler, step)
    





    