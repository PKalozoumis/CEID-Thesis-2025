from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
import time
import requests
import subprocess
from rich.console import Console
from matplotlib import pyplot as plt

console = Console()

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

x = []
y = []

for i in range (33):
    proc = subprocess.Popen(["llama-server", "--device", "CUDA0", "--gpu-layers", str(i)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    mem = 0

    while True:
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        try:
            resp = requests.get("http://localhost:8080/health")
            resp.raise_for_status()
            x.append(i)
            mem = nvmlDeviceGetMemoryInfo(handle).used
            y.append(mem)
            break
        except BaseException:
            #console.print("[red]billkourt error[/red]")
            time.sleep(1)

    console.print(f"Offloaded {i:02}/33: [green]{mem // (1024**2)} / {mem_info.total // (1024**2)} MB[/green]")
    proc.terminate()
    time.sleep(2)

fig, ax = plt.subplots()
ax.plot(x,y)
plt.show(block=True)

nvmlShutdown()  # Cleanup