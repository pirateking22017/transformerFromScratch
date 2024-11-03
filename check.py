import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the current device
    device = torch.cuda.current_device()
    
    # Get the total and allocated memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)

    print(f"Total Memory: {total_memory / (1024 ** 2):.2f} MB")
    print(f"Allocated Memory: {allocated_memory / (1024 ** 2):.2f} MB")
    print(f"Reserved Memory: {reserved_memory / (1024 ** 2):.2f} MB")
else:
    print("CUDA is not available.")
