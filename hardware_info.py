import subprocess
import psutil
import re

def get_mig_gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout

        # Match lines like: "|  0    1   0   0  |   34225MiB / 40192MiB  |"
        mig_mem_lines = re.findall(r'\|\s+\d+\s+\d+\s+\d+\s+\d+\s+\|\s+(\d+)MiB\s+/\s+(\d+)MiB', output)

        total_mem_mib = sum(int(total) for _, total in mig_mem_lines)
        num_mig_devices = len(mig_mem_lines)
        total_mem_gb = total_mem_mib / 1024
        return num_mig_devices, total_mem_gb

    except Exception as e:
        return 0, 0.0

def get_cpu_info():
    cores = psutil.cpu_count(logical=False)
    threads = psutil.cpu_count(logical=True)
    freq = psutil.cpu_freq()
    return cores, threads, round(freq.max / 1000, 2)

# Get data
num_mig_devices, total_gpu_gb = get_mig_gpu_memory()
cores, threads, freq_ghz = get_cpu_info()

# Display
print(f"Number of MIG Devices: {num_mig_devices}")
print(f"Total GPU Memory (GB): {total_gpu_gb:.2f}")
print(f"CPU Cores (Physical): {cores}")
print(f"CPU Threads (Logical): {threads}")
print(f"CPU Max Frequency: {freq_ghz:.2f} GHz")
