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
            elif 2048 == sequence_length:  # 这个真的只能是2，不然显存不够，使用fp16，然后adafactor 也不能提升到3
                batch_size = 2
            elif 4096 == sequence_length:
                batch_size = 1
            else:
                raise ValueError("sequence_length error")
        elif "reformer" and "gpt" in model_type:
            if 2048 == sequence_length:
                batch_size = 2
            else:
                raise ValueError("sequence_length error")
        elif "reformer" in model_type:
            if 1024 == sequence_length:
                batch_size = 16
            elif 8196 == sequence_length:
                batch_size = 8
            else:
                raise ValueError("sequence_length error")
        elif "retnet" in model_type:
            if 512 == sequence_length:
                batch_size = 16
            elif 1024 == sequence_length:
                batch_size = 8
            elif 2048 == sequence_length:
                # batch_size = 4
                batch_size = 2
            else:
                raise ValueError("sequence_length error")
        else:
            raise ValueError("model_type error")
    elif model_size == "large":
        if "reformer" in model_type and "gpt" in model_type:
            if 2048 == sequence_length:
                batch_size = 2
            else:
                raise ValueError("sequence_length error")
        elif "reformer" in model_type:
            if 1024 == sequence_length:
                batch_size = 8
            elif 2048 == sequence_length:
                batch_size = 4
            else:
                raise ValueError("sequence_length error")
        elif "retnet" in model_type:
            if 1024 == sequence_length:
                batch_size = 4
            elif 2048 == sequence_length:
                batch_size = 2
            else:
                raise ValueError("sequence_length error")
    
    return batch_size