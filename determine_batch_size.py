def get_batch_size(model_size : str,model_type:str,sequence_length:int):  # 这个可能还要考虑SGD的影响
    if model_size == "base":
        if model_type == "bert":
            if 128 == sequence_length:
                batch_size = 64
            elif 512 == sequence_length:
                batch_size = 16
            elif 2048 == sequence_length:
                batch_size = 4
        elif True:
            raise ValueError("model_type error")
    elif True:
        raise ValueError("model_size error")
    
    return batch_size