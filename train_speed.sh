#!/bin/bash
#SBATCH -o job.%j-speed.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=gpulab02      
#SBATCH --qos=gpulab02             # 指定作业的QOS
#SBATCH -J zly       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=12    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:2           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 
#SBATCH --nodelist=gpu030
nvidia-smi
accelerate launch --config_file acce_config_1GPU.yaml --main_process_port 29501 train_vanilla_bert.py
accelerate launch --config_file acce_config_1GPU.yaml --main_process_port 29501 train_cos_bert.py
accelerate launch --config_file acce_config_1GPU.yaml --main_process_port 29501 train_directmul_bert.py