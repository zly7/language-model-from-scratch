def get_batch_size(model_size : str,model_type:str,sequence_length:int,use_cos:bool = False, use_reformer:bool= False):  # 这个可能还要考虑SGD的影响
    if model_size == "base":
        if model_type == "bert":
            if 128 == sequence_length:
                batch_size = 64
            elif 512 == sequence_length:
                batch_size = 16
            elif 1024 == sequence_length:
                batch_size = 8
                # if use_reformer is True:
                #     batch_size = 16
                    # batch_size = 12
            elif 2048 == sequence_length:
                batch_size = 2
            elif 4096 == sequence_length:
                batch_size = 1
        elif "gpt" in model_type:
            if 128 == sequence_length:
                batch_size = 60
            elif 512 == sequence_length:
                batch_size = 12
                if use_cos is True:
                    batch_size = 4
            elif 1024 == sequence_length:
                batch_size = 6
            elif 2048 == sequence_length:
                batch_size = 2
            elif 4096 == sequence_length:
                batch_size = 1
        elif "reformer" in model_type:
            if 1024 == sequence_length:
                batch_size = 16
        else:
            raise ValueError("model_type error")
    elif True:
        raise ValueError("model_size error")
    
    return batch_size