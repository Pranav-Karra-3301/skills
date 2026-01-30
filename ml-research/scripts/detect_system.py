#!/usr/bin/env python3
"""
ML System Detection Script

Detects and reports compute environment for ML work:
- GPU (NVIDIA), CUDA, cuDNN versions
- Memory (RAM and VRAM)
- Python version, virtual environment
- Installed ML frameworks and tools
- Estimated capacity for common model sizes
"""

import subprocess
import sys
import os
import platform
from pathlib import Path


def run_cmd(cmd: str, shell: bool = True) -> str:
    """Run command and return output, empty string on failure."""
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_python_info() -> dict:
    """Get Python environment information."""
    info = {
        "version": platform.python_version(),
        "executable": sys.executable,
        "virtual_env": None,
        "conda_env": None,
    }

    if os.environ.get("VIRTUAL_ENV"):
        info["virtual_env"] = os.environ["VIRTUAL_ENV"]
    if os.environ.get("CONDA_DEFAULT_ENV"):
        info["conda_env"] = os.environ["CONDA_DEFAULT_ENV"]

    return info


def get_gpu_info() -> dict:
    """Get NVIDIA GPU information."""
    info = {
        "available": False,
        "gpus": [],
        "cuda_version": None,
        "cudnn_version": None,
        "driver_version": None,
    }

    # Check nvidia-smi
    nvidia_output = run_cmd(
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits"
    )
    if nvidia_output:
        info["available"] = True
        for line in nvidia_output.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                info["gpus"].append({
                    "name": parts[0],
                    "memory_mb": int(parts[1]) if parts[1].isdigit() else parts[1],
                    "memory_gb": round(int(parts[1]) / 1024, 1) if parts[1].isdigit() else None,
                })
                info["driver_version"] = parts[2]

    # Check CUDA version
    nvcc_output = run_cmd("nvcc --version")
    if nvcc_output and "release" in nvcc_output:
        for line in nvcc_output.split("\n"):
            if "release" in line:
                # Parse "Cuda compilation tools, release 11.8, V11.8.89"
                parts = line.split("release")
                if len(parts) > 1:
                    version = parts[1].split(",")[0].strip()
                    info["cuda_version"] = version

    # Try to get CUDA version from PyTorch
    if not info["cuda_version"]:
        try:
            import torch
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
        except ImportError:
            pass

    return info


def get_memory_info() -> dict:
    """Get system memory information."""
    info = {"ram_gb": None, "ram_available_gb": None}

    # Linux/WSL
    if platform.system() == "Linux":
        meminfo = run_cmd("free -b")
        if meminfo:
            lines = meminfo.split("\n")
            for line in lines:
                if line.startswith("Mem:"):
                    parts = line.split()
                    if len(parts) >= 7:
                        info["ram_gb"] = round(int(parts[1]) / 1e9, 1)
                        info["ram_available_gb"] = round(int(parts[6]) / 1e9, 1)

    # macOS
    elif platform.system() == "Darwin":
        memsize = run_cmd("sysctl -n hw.memsize")
        if memsize:
            info["ram_gb"] = round(int(memsize) / 1e9, 1)

    return info


def get_ml_frameworks() -> dict:
    """Detect installed ML frameworks and their GPU support."""
    frameworks = {}

    # PyTorch
    try:
        import torch
        frameworks["pytorch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            frameworks["pytorch"]["devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    # TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        frameworks["tensorflow"] = {
            "version": tf.__version__,
            "gpu_count": len(gpus),
            "gpu_available": len(gpus) > 0,
        }
    except ImportError:
        pass

    # JAX
    try:
        import jax
        frameworks["jax"] = {
            "version": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
            "default_backend": jax.default_backend(),
        }
    except ImportError:
        pass

    return frameworks


def get_ml_tools() -> dict:
    """Detect installed ML tools."""
    tools = {}

    # Weights & Biases
    try:
        import wandb
        tools["wandb"] = {
            "version": wandb.__version__,
            "logged_in": bool(run_cmd("wandb status 2>/dev/null | grep 'Logged in'")),
        }
    except ImportError:
        pass

    # MLflow
    try:
        import mlflow
        tools["mlflow"] = {
            "version": mlflow.__version__,
            "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "not set"),
        }
    except ImportError:
        pass

    # Hydra
    try:
        import hydra
        tools["hydra"] = {"version": hydra.__version__}
    except ImportError:
        pass

    # Transformers (Hugging Face)
    try:
        import transformers
        tools["transformers"] = {"version": transformers.__version__}
    except ImportError:
        pass

    # PEFT
    try:
        import peft
        tools["peft"] = {"version": peft.__version__}
    except ImportError:
        pass

    # bitsandbytes (for quantization)
    try:
        import bitsandbytes
        tools["bitsandbytes"] = {"version": bitsandbytes.__version__}
    except ImportError:
        pass

    return tools


def estimate_model_capacity(gpu_memory_gb: float) -> dict:
    """Estimate what models can fit on available GPU memory."""
    if not gpu_memory_gb:
        return {}

    # Rough estimates for training (params + grads + optimizer states)
    # fp16 training: ~4 bytes per parameter (param + grad) + optimizer states
    # With Adam: ~16 bytes per parameter total for training

    estimates = {
        "inference_fp16": {
            "7B": gpu_memory_gb >= 14,
            "13B": gpu_memory_gb >= 26,
            "70B": gpu_memory_gb >= 140,
        },
        "inference_4bit": {
            "7B": gpu_memory_gb >= 4,
            "13B": gpu_memory_gb >= 8,
            "70B": gpu_memory_gb >= 40,
        },
        "finetune_lora": {
            "7B": gpu_memory_gb >= 16,
            "13B": gpu_memory_gb >= 32,
        },
        "finetune_qlora": {
            "7B": gpu_memory_gb >= 8,
            "13B": gpu_memory_gb >= 16,
            "70B": gpu_memory_gb >= 48,
        },
    }

    return estimates


def print_report():
    """Print comprehensive system report."""
    print("=" * 60)
    print("ML SYSTEM DETECTION REPORT")
    print("=" * 60)

    # Python
    print("\n[Python Environment]")
    py_info = get_python_info()
    print(f"  Version: {py_info['version']}")
    print(f"  Executable: {py_info['executable']}")
    if py_info["virtual_env"]:
        print(f"  Virtual Env: {py_info['virtual_env']}")
    if py_info["conda_env"]:
        print(f"  Conda Env: {py_info['conda_env']}")

    # Memory
    print("\n[System Memory]")
    mem_info = get_memory_info()
    if mem_info["ram_gb"]:
        print(f"  Total RAM: {mem_info['ram_gb']} GB")
        if mem_info["ram_available_gb"]:
            print(f"  Available RAM: {mem_info['ram_available_gb']} GB")
    else:
        print("  Could not detect system memory")

    # GPU
    print("\n[GPU Information]")
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        for i, gpu in enumerate(gpu_info["gpus"]):
            print(f"  GPU {i}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_gb']} GB")
        print(f"  Driver: {gpu_info['driver_version']}")
        if gpu_info["cuda_version"]:
            print(f"  CUDA: {gpu_info['cuda_version']}")
    else:
        print("  No NVIDIA GPU detected")
        print("  Training will run on CPU (significantly slower)")

    # ML Frameworks
    print("\n[ML Frameworks]")
    frameworks = get_ml_frameworks()
    if not frameworks:
        print("  No ML frameworks detected")
        print("  Install: pip install torch  # or tensorflow")
    else:
        for name, info in frameworks.items():
            cuda_status = ""
            if name == "pytorch":
                cuda_status = f", CUDA: {'Yes' if info['cuda_available'] else 'No'}"
            elif name == "tensorflow":
                cuda_status = f", GPU: {'Yes' if info['gpu_available'] else 'No'}"
            print(f"  {name}: {info['version']}{cuda_status}")

    # ML Tools
    print("\n[ML Tools]")
    tools = get_ml_tools()
    if not tools:
        print("  No ML tools detected")
        print("  Recommended: pip install wandb hydra-core")
    else:
        for name, info in tools.items():
            extra = ""
            if name == "wandb":
                extra = f" ({'logged in' if info['logged_in'] else 'not logged in'})"
            print(f"  {name}: {info['version']}{extra}")

    # Capacity Estimates
    if gpu_info["available"] and gpu_info["gpus"]:
        total_gpu_memory = sum(g["memory_gb"] for g in gpu_info["gpus"] if g["memory_gb"])
        if total_gpu_memory:
            print("\n[Model Capacity Estimates]")
            estimates = estimate_model_capacity(total_gpu_memory)

            print(f"  Total GPU Memory: {total_gpu_memory} GB")
            print("\n  Inference (4-bit quantized):")
            for size, fits in estimates.get("inference_4bit", {}).items():
                status = "✓" if fits else "✗"
                print(f"    {status} {size} model")

            print("\n  Finetuning (QLoRA):")
            for size, fits in estimates.get("finetune_qlora", {}).items():
                status = "✓" if fits else "✗"
                print(f"    {status} {size} model")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_report()
