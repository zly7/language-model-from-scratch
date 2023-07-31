import torch

def test_torch_and_cuda():
    # Check if PyTorch is installed successfully
    print("PyTorch version:", torch.__version__)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available.")
        print("GPU device name:", torch.cuda.get_device_name(0))  # Get the name of the first GPU device
    else:
        print("CUDA is not available. Only CPU will be used for computations.")

if __name__ == "__main__":
    test_torch_and_cuda()
