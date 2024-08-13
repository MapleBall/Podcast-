import torch

if torch.cuda.is_available():
    print("GPU is available. Using GPU for inference.")
else:
    print("GPU is not available. Using CPU for inference.")
import psutil

# 獲取 CPU 核心數量
logical_cores = psutil.cpu_count(logical=True)
physical_cores = psutil.cpu_count(logical=False)
print(f"邏輯核心數量: {logical_cores}")
print(f"實際核心數量: {physical_cores}")

# 獲取 CPU 相關信息
cpu_freq = psutil.cpu_freq()
print(f"當前頻率: {cpu_freq.current} MHz")
print(f"最低頻率: {cpu_freq.min} MHz")
print(f"最高頻率: {cpu_freq.max} MHz")

# 獲取 CPU 使用率
cpu_usage = psutil.cpu_percent(interval=1)
print(f"當前 CPU 使用率: {cpu_usage}%")
