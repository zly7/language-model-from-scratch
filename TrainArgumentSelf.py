from transformers import TrainingArguments

class TrainingArgumentsSelf(TrainingArguments):
    def __init__(self, test_step = None, per_device_test_batch_size = 32, all_test_examples_num = 256, train_audit_probability=0, test_dataloader_use_accelerate=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_step = test_step
        self.train_audit_probability = train_audit_probability
        self.test_dataloader_use_accelerate = test_dataloader_use_accelerate
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

            self.per_device_test_batch_size = per_device_test_batch_size
            self.all_test_examples_num = all_test_examples_num
        

