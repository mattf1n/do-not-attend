import torch, os

print(f"Host: {os.uname().nodename}")
print(f"CPUs: {os.cpu_count()}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
print(f"Using: {'cuda' if torch.cuda.is_available() else 'cpu'}")
