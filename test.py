import torch

if torch.cuda.is_available():
    print(f"Visible GPU Index: {torch.cuda.current_device()}")  # Prints the GPU index
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("No GPU detected.")
