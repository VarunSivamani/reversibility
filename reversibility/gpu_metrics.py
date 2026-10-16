"""NVIDIA GPU telemetry via NVML (pynvml), with a graceful no-op fallback
when NVML/pynvml/a GPU isn't available, so callers (train.py) can log these
metrics unconditionally without special-casing CPU/MPS/Mac dry-runs.
"""

from __future__ import annotations

import torch

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:  # noqa: BLE001 - any failure (no pynvml, no driver, no GPU) just disables telemetry
    _NVML_OK = False


def nvml_available() -> bool:
    return _NVML_OK


def gpu_snapshot(device_index: int = 0) -> dict[str, float]:
    """One point-in-time snapshot of GPU telemetry plus torch's own CUDA
    allocator stats. Returns NaN for any field that can't be read (no NVML,
    no CUDA, a metric this GPU/driver doesn't expose)."""

    metrics: dict[str, float] = {
        "gpu_util_pct": float("nan"),
        "mem_controller_util_pct": float("nan"),
        "power_draw_w": float("nan"),
        "temperature_c": float("nan"),
        "sm_clock_mhz": float("nan"),
        "mem_clock_mhz": float("nan"),
        "cuda_mem_allocated_mb": float("nan"),
        "cuda_mem_reserved_mb": float("nan"),
        "cuda_max_mem_allocated_mb": float("nan"),
        "cuda_max_mem_reserved_mb": float("nan"),
    }

    if torch.cuda.is_available():
        metrics["cuda_mem_allocated_mb"] = torch.cuda.memory_allocated(device_index) / 1e6
        metrics["cuda_mem_reserved_mb"] = torch.cuda.memory_reserved(device_index) / 1e6
        metrics["cuda_max_mem_allocated_mb"] = torch.cuda.max_memory_allocated(device_index) / 1e6
        metrics["cuda_max_mem_reserved_mb"] = torch.cuda.max_memory_reserved(device_index) / 1e6

    if not _NVML_OK:
        return metrics

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics["gpu_util_pct"] = float(util.gpu)
        metrics["mem_controller_util_pct"] = float(util.memory)

        try:
            metrics["power_draw_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except pynvml.NVMLError:  # noqa: BLE001 - not every card/driver exposes power
            pass

        try:
            metrics["temperature_c"] = float(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except pynvml.NVMLError:  # noqa: BLE001
            pass

        try:
            metrics["sm_clock_mhz"] = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            )
            metrics["mem_clock_mhz"] = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            )
        except pynvml.NVMLError:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001 - never let telemetry collection crash training
        pass

    return metrics


def gpu_name(device_index: int = 0) -> str:
    if not _NVML_OK:
        return "unknown"
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(handle)
        return name.decode() if isinstance(name, bytes) else name
    except Exception:  # noqa: BLE001
        return "unknown"
