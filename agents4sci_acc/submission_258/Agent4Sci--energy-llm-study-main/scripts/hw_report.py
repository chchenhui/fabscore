import os
import json
import platform
import psutil

report = {
    "platform": platform.platform(),
    "cpu": platform.processor(),
    "cores": psutil.cpu_count(logical=True),
    "memory_gb": round(psutil.virtual_memory().total / 1e9, 2),
    "openblas_threads": os.getenv("OPENBLAS_NUM_THREADS"),
    "mkl_threads": os.getenv("MKL_NUM_THREADS"),
    "omp_threads": os.getenv("OMP_NUM_THREADS")
}

os.makedirs("docs", exist_ok=True)
with open("docs/SYSTEM_REPORT.md", "w") as f:
    f.write("```json\n")
    json.dump(report, f, indent=2)
    f.write("\n```")

print("✅ Hardware report saved to docs/SYSTEM_REPORT.md")
