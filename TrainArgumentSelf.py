from transformers import TrainingArguments

from dataclasses import dataclass, field

@dataclass
class TrainingArgumentsSelf(TrainingArguments):
    # def __init__(self, test_step = None, per_device_test_batch_size = 32, all_test_examples_num = 256,
    #  train_audit_probability=0, test_dataloader_use_accelerate=False, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.test_step = test_step
    #     self.train_audit_probability = train_audit_probability
    #     self.test_dataloader_use_accelerate = test_dataloader_use_accelerate
    #     if self.max_steps is not None:
    #         self.num_train_epochs = 10000
    #         Warning("max_steps is not None, so num_train_epochs will be 10000, which is useless")
    #     elif self.num_train_epochs is None:
    #         self.max_steps = 1e9
    #         Warning("num_train_epochs is None, so max_steps will be 1e9, which is useless")
    #     if self.test_step is None:
    #         self.per_device_test_batch_size = None
    #         self.all_test_examples_num = None
    #         Warning("test_step is None, so per_device_test_batch_size and all_test_examples_num will be None")
    #     else:

    #         self.per_device_test_batch_size = per_device_test_batch_size
    #         self.all_test_examples_num = all_test_examples_num
    # gradient_ac:int = field(default = None)  # 这个参数有
    test_step:int = field(default=None)
    train_audit_probability:float = field(default=0.0)
    test_dataloader_use_accelerate:bool = field(default=True)
    per_device_test_batch_size:int = field(default=32)    # test data set batch
    all_test_examples_num:int = field(default=256)
    whether_hg_accelerator:bool = field(default=True)
    optimizer_type:str = field(default="adam")
    sgd_momentum:float = field(default=0.9)
    sequence_length:int = field(default=512)
    data_set_path:str = field(default="this is to describe the used dataset")

    def __post_init__(self):
        super().__post_init__()
        # if self.gradient_ac == None:
        #     raise ValueError("TrainingArgumentsSelf Must record gradient_ac")
        if self.max_steps is not None:
            self.num_train_epochs = 10000
            Warning("max_steps is not None, so num_train_epochs will be 10000, which is useless")
        elif self.num_train_epochs is None:
            self.max_steps = 1e9
            Warning("num_train_epochs is None, so max_steps will be 1e9, which is useless")
        if self.test_step is None:
            self.per_device_test_batch_size = None
            self.all_test_examples_num = None
            Warning("test_step is None, so per_device_test_batch_size and all_test_examples_num will be None")
        else:
            pass




    def to_dict(self):
        return super().to_dict()
        

