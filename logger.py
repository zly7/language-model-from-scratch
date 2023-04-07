from torch.utils.tensorboard import SummaryWriter
class Logger:
    def __init__(self, logger:str, output_path:str) -> None:
        if logger == "tensorboard":
            self.logger = SummaryWriter(log_dir=output_path)
        else:
            self.logger = SummaryWriter(log_dir=output_path)

    def log_scaler(self,path, number, step):
        self.logger.add_scalar(path,number,step)
